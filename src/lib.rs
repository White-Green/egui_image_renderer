use egui::{Context, Pos2, RawInput, Rect};
use egui_wgpu::renderer::ScreenDescriptor;
use egui_wgpu::{wgpu::*, Renderer};
use futures::FutureExt;
use image::{ImageError, RgbaImage};
use std::future::Future;
use std::io::{Seek, Write};
use thiserror::Error;
use tokio::sync::oneshot;

#[derive(Debug, Error)]
pub enum EguiImageRendererError {
    #[error("no wgpu adapter found")]
    AdapterNotFound,
    #[error("{0}")]
    Wgpu(#[from] RequestDeviceError),
    #[error("{0}")]
    ImageError(#[from] ImageError),
}

struct EguiRenderContext {
    device: Device,
    queue: Queue,
}

impl EguiRenderContext {
    async fn new() -> Result<EguiRenderContext, EguiImageRendererError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .ok_or(EguiImageRendererError::AdapterNotFound)?;

        let (device, queue) = adapter.request_device(&Default::default(), None).await?;

        Ok(EguiRenderContext { device, queue })
    }

    fn render_into_texture_and_buffer(
        &self,
        ui: impl FnOnce(&Context),
        texture: &Texture,
        buffer: &Buffer,
    ) -> impl Future<Output = ()> + Send {
        assert_eq!(texture.format(), TextureFormat::Rgba8Unorm);
        assert_eq!(
            buffer.size(),
            texture.width() as u64 * texture.height() as u64 * 4
        );
        let ctx = Context::default();
        let EguiRenderContext { device, queue } = self;
        let mut renderer = Renderer::new(&device, TextureFormat::Rgba8Unorm, None, 1);

        let output = ctx.run(
            RawInput {
                screen_rect: Some(Rect {
                    min: Pos2::ZERO,
                    max: Pos2::new(texture.width() as f32, texture.height() as f32),
                }),
                ..Default::default()
            },
            ui,
        );
        for (id, delta) in output.textures_delta.set {
            renderer.update_texture(&device, &queue, id, &delta);
        }
        let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);

        let mut encoder = device.create_command_encoder(&Default::default());
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [texture.width(), texture.height()],
            pixels_per_point: output.pixels_per_point,
        };
        renderer.update_buffers(device, queue, &mut encoder, &primitives, &screen_descriptor);
        let texture_view = texture.create_view(&Default::default());
        let color_attachments = &[Some(RenderPassColorAttachment {
            view: &texture_view,
            resolve_target: None,
            ops: Default::default(),
        })];
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                color_attachments,
                ..Default::default()
            });
            renderer.render(&mut rpass, &primitives, &screen_descriptor);
        }

        buffer.unmap();
        let bytes_per_row = ceil_256(texture.width() as u64 * 4);
        let buffer_dst = ImageCopyBuffer {
            buffer: &buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row as u32),
                rows_per_image: None,
            },
        };
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            buffer_dst,
            Extent3d {
                width: texture.width(),
                height: texture.height(),
                depth_or_array_layers: 1,
            },
        );

        let (sender, receiver) = oneshot::channel();
        queue.on_submitted_work_done(|| sender.send(()).unwrap());
        queue.submit([encoder.finish()]);
        receiver.map(Result::unwrap)
    }

    fn create_texture_and_buffer(&self, width: u32, height: u32) -> (Texture, Buffer) {
        let texture = self.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: ceil_256(width as u64 * 4) * height as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: true,
        });
        (texture, buffer)
    }
}

fn ceil_256(x: u64) -> u64 {
    (x + 0xff) & !0xffu64
}

pub enum FileFormat {
    PNG,
    JPEG,
}

static EGUI_RENDER_CONTEXT: tokio::sync::OnceCell<EguiRenderContext> =
    tokio::sync::OnceCell::const_new();

pub async fn render_into_file(
    ui: impl FnOnce(&Context),
    width: u32,
    height: u32,
    format: FileFormat,
    mut out: impl Write + Seek,
) -> Result<(), EguiImageRendererError> {
    let context = EGUI_RENDER_CONTEXT
        .get_or_try_init(EguiRenderContext::new)
        .await?;
    let (texture, buffer) = context.create_texture_and_buffer(width, height);
    context
        .render_into_texture_and_buffer(ui, &texture, &buffer)
        .await;

    let bslice = buffer.slice(..);
    let (tx, rx) = oneshot::channel();
    bslice.map_async(MapMode::Read, |_| tx.send(()).unwrap_or(()));
    context.device.poll(Maintain::Wait);
    rx.await.ok();
    let data = bslice.get_mapped_range();

    let mut img = RgbaImage::new(width, height);
    for (img, data) in img
        .chunks_mut(width as usize * 4)
        .zip(data.chunks(ceil_256(width as u64 * 4) as usize))
    {
        assert!(img.len() <= data.len());
        img.copy_from_slice(&data[..img.len()]);
    }
    img.copy_from_slice(&data);
    img.write_to(
        &mut out,
        match format {
            FileFormat::PNG => image::ImageFormat::Png,
            FileFormat::JPEG => image::ImageFormat::Jpeg,
        },
    )
    .map_err(Into::into)
}
