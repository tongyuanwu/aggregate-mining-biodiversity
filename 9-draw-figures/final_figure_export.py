"""Export publication figures with a checked physical width and editable text.

Requires pypdf in addition to the figure notebooks' existing dependencies.
The working canvas can be larger than the final figure. Only the complete
PDF page is scaled, so maps and plots retain their aspect ratios.
"""
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
from matplotlib.text import Text
from pypdf import PdfReader, PdfWriter


def export_final_figure(fig, stem, max_width_mm=180.0, jpg_dpi=600):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Preserve the established panel layout while adjusting typography.
    fig.set_layout_engine('none')
    renderer = fig.canvas.get_renderer()
    initial_bbox = fig.get_tightbbox(renderer).padded(0.1)
    initial_scale = min(1.0, max_width_mm / (initial_bbox.width * 25.4))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    labels = [t for t in texts if t.get_text() in tuple('abcdefghijklmnopqrstuvwxyz')
              and t.get_fontsize() >= 11]
    desired_sizes = {
        t: 8.0 if t in labels else min(7.0, max(5.0, t.get_fontsize() * initial_scale))
        for t in texts
    }
    scale = initial_scale
    # Font changes slightly alter the cropped page width. Resolve that feedback
    # before exporting so both the physical width and final point sizes are correct.
    for _ in range(12):
        for text, final_size in desired_sizes.items():
            text.set_fontsize(final_size / scale)
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        next_scale = min(1.0, max_width_mm / (bbox.width * 25.4))
        if abs(next_scale - scale) < 1e-8:
            break
        scale = next_scale
    else:
        raise RuntimeError('Final figure size did not converge')

    pdf_buffer = BytesIO()
    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig.savefig(pdf_buffer, format='pdf', dpi=600, bbox_inches=bbox)
    pdf_buffer.seek(0)
    reader = PdfReader(pdf_buffer)
    page = reader.pages[0]
    scale = min(1.0, (max_width_mm / 25.4 * 72) / float(page.mediabox.width))
    page.scale_by(scale)
    writer = PdfWriter()
    writer.add_page(page)
    writer.pages[0].compress_content_streams()
    writer.add_metadata({'/Title': stem.name,
                         '/Subject': f'Maximum width: {max_width_mm:g} mm; editable TrueType text'})
    with stem.with_suffix('.pdf').open('wb') as output:
        writer.write(output)
    for extension, dpi in [('png', 300), ('jpg', jpg_dpi)]:
        options = {'dpi': (dpi, dpi)}
        if extension == 'jpg':
            options['quality'] = 95
        fig.savefig(stem.with_suffix('.' + extension), dpi=dpi * scale,
                    bbox_inches=bbox, pil_kwargs=options)

    delivered = PdfReader(stem.with_suffix('.pdf')).pages[0]
    width_mm = float(delivered.mediabox.width) * 25.4 / 72
    height_mm = float(delivered.mediabox.height) * 25.4 / 72
    assert width_mm <= max_width_mm + 1e-6
    assert all(font.get_object()['/Subtype'] != '/Type3'
               for font in delivered['/Resources']['/Font'].get_object().values())
    standard_sizes = [t.get_fontsize() * scale for t in texts if t not in labels]
    assert min(standard_sizes) >= 5.0 - 1e-5
    assert max(standard_sizes) <= 7.0 + 1e-5
    return {'width_mm': width_mm, 'height_mm': height_mm,
            'standard_text_pt': [min(standard_sizes), max(standard_sizes)],
            'panel_label_pt': [t.get_fontsize() * scale for t in labels],
            'scale': scale, 'type3_fonts': False}
