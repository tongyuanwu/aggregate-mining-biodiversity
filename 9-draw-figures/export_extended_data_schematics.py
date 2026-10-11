"""Fit the vector schematics to 180 mm, preserving all artwork proportions.

ED1 retains its complete layout. ED4 relocates its six-item right-hand key
below the diagram, in two columns, to make the diagram and labels larger.
Original PowerPoint-derived PDF artwork is retained under source data.
Requires pypdf, pymupdf and Pillow.
"""
from pathlib import Path
import argparse
import json

import pymupdf
from pypdf import PdfReader, PdfWriter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source data/figure artwork'
PT_PER_MM = 72 / 25.4


def raster_copies(path):
    with pymupdf.open(path) as doc:
        pixmap = doc[0].get_pixmap(dpi=300, alpha=False)
        image = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
        image.save(path.with_suffix('.png'), dpi=(300, 300))
        image.save(path.with_suffix('.jpg'), dpi=(300, 300), quality=95)


def export_schematics(out):
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    source1 = SOURCE / 'Extended-Data-Figure-1-original.pdf'
    reader = PdfReader(source1)
    page = reader.pages[0]
    original_width, original_height = float(page.mediabox.width), float(page.mediabox.height)
    scale = 180 * PT_PER_MM / original_width
    page.scale_by(scale)
    writer = PdfWriter()
    writer.add_page(page)
    writer.pages[0].compress_content_streams()
    destination = out / 'Extended-Data-Figure-1.pdf'
    with destination.open('wb') as output:
        writer.write(output)
    raster_copies(destination)
    reports.append({'figure': 1, 'width_mm': 180.0,
                    'height_mm': original_height * scale / PT_PER_MM,
                    'uniform_scale': scale, 'aspect_ratio_preserved': True,
                    'layout': 'Original complete flowchart retained'})

    source4 = pymupdf.open(SOURCE / 'Extended-Data-Figure-4-original.pdf')
    # Coordinates refer to the reviewed 1400-pixel-wide proof of the source page.
    units = source4[0].rect.width / 1400
    def clip(coords):
        return pymupdf.Rect(*(value * units for value in coords))

    core = clip((18, 12, 1028, 798))
    first_key = clip((1067, 393, 1378, 498))
    second_key = clip((1067, 504, 1378, 599))
    key_header = clip((1067, 393, 1378, 413))
    margin = 2 * PT_PER_MM
    scale = (180 * PT_PER_MM - 2 * margin) / core.width
    core_height = core.height * scale
    key_y = margin + core_height + 4 * PT_PER_MM
    key_height = max(first_key.height, key_header.height + second_key.height) * scale
    final_height = key_y + key_height + margin
    result = pymupdf.open()
    page = result.new_page(width=180 * PT_PER_MM, height=final_height)

    def place(region, x, y):
        target = pymupdf.Rect(x, y, x + region.width * scale, y + region.height * scale)
        page.show_pdf_page(target, source4, 0, clip=region, keep_proportion=True)

    place(core, margin, margin)
    # Use the same scale for every piece of artwork and every legend item.
    first_x, second_x = 29 * PT_PER_MM, 104 * PT_PER_MM
    place(first_key, first_x, key_y)
    place(key_header, second_x, key_y)
    place(second_key, second_x, key_y + key_header.height * scale)
    destination = out / 'Extended-Data-Figure-4.pdf'
    result.set_metadata({'title': 'Extended Data Figure 4',
                         'subject': '180 mm width; original vector artwork; key moved below diagram'})
    result.save(destination, garbage=4, deflate=True)
    result.close()
    source4.close()
    raster_copies(destination)
    reports.append({'figure': 4, 'width_mm': 180.0,
                    'height_mm': final_height / PT_PER_MM,
                    'uniform_scale': scale, 'aspect_ratio_preserved': True,
                    'layout': 'Original diagram; six-item key moved below in two columns'})
    for report in reports:
        (out / f'ED-{report["figure"]}-verification.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    print(json.dumps(export_schematics(args.output_dir), indent=2))
