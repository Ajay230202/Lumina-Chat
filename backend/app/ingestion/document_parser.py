from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

class DocumentParser:
    def __init__(self):
        pipeline_opts = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            generate_page_images=False,
        )
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )

    def parse(self, file_path: str) -> dict:
        result = self.converter.convert(file_path)
        doc = result.document

        num_pages = len(doc.pages) if doc.pages else 1

        # Group text elements by page number
        pages_content = {i: [] for i in range(1, num_pages + 1)}

        for item in doc.texts:
            for p in item.prov:
                if p.page_no in pages_content:
                    pages_content[p.page_no].append(item.text)

        # Build page schemas
        pages = []
        for page_num in range(1, num_pages + 1):
            # Extract tables on this page
            tables = []
            for t in doc.tables:
                t_page = t.prov[0].page_no if t.prov else 1
                if t_page == page_num:
                    # In newer docling versions, tables export to markdown via a helper
                    # or directly from their data representation
                    table_md = ""
                    try:
                        table_md = t.export_to_markdown()
                    except Exception:
                        pass
                    tables.append({
                        "markdown": table_md,
                        "caption": getattr(t, "caption", "") or ""
                    })
                    # Append table markdown to page text representation as well
                    if table_md:
                        pages_content[page_num].append(table_md)

            pages.append({
                "page_num": page_num,
                "text": "\n\n".join(pages_content[page_num]),
                "tables": tables,
                "image_refs": []
            })

        return {
            "text_markdown": doc.export_to_markdown(),
            "pages": pages,
            "metadata": {
                "title": Path(file_path).stem,
                "num_pages": num_pages,
                "source_path": file_path,
            },
        }
