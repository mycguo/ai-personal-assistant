import mammoth from "mammoth";
import ExcelJS from "exceljs";

export async function extractTextFromFile(file: File) {
  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  const name = file.name;
  const extension = name.split(".").pop()?.toLowerCase() ?? "";

  if (["txt", "md", "markdown", "csv", "json", "html", "htm"].includes(extension)) {
    return buffer.toString("utf8");
  }

  if (extension === "docx") {
    const result = await mammoth.extractRawText({ buffer });
    return result.value;
  }

  if (extension === "xlsx") {
    const workbook = new ExcelJS.Workbook();
    await workbook.xlsx.load(arrayBuffer as Parameters<typeof workbook.xlsx.load>[0]);
    return workbook.worksheets
      .map((worksheet) => {
        const rows: string[] = [];
        worksheet.eachRow((row) => {
          const values = Array.isArray(row.values) ? row.values.slice(1) : [];
          rows.push(values.map((value) => String(value ?? "")).join(","));
        });
        return `Sheet: ${worksheet.name}\n${rows.join("\n")}`;
      })
      .join("\n\n");
  }

  if (extension === "pdf") {
    const { PDFParse } = await import("pdf-parse");
    const parser = new PDFParse({ data: buffer });
    try {
      const result = await parser.getText();
      return result.text;
    } finally {
      await parser.destroy();
    }
  }

  throw new Error(`Unsupported file type: .${extension || "unknown"}`);
}
