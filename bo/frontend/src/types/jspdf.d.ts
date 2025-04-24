declare module 'jspdf' {
  export interface JsPDFOptions {
    orientation?: 'portrait' | 'landscape';
    unit?: 'pt' | 'mm' | 'cm' | 'in';
    format?: string | [number, number];
    compress?: boolean;
    precision?: number;
    filters?: string[];
  }

  export class jsPDF {
    constructor(options?: JsPDFOptions);
    addImage(
      imageData: string | HTMLImageElement | HTMLCanvasElement,
      format: string,
      x: number,
      y: number,
      width: number,
      height: number,
      alias?: string,
      compression?: string,
      rotation?: number
    ): jsPDF;
    save(filename: string): jsPDF;
    output(type: string, options?: any): any;
    setProperties(properties: any): jsPDF;
    setFontSize(size: number): jsPDF;
    setFont(fontName: string, fontStyle?: string): jsPDF;
    setTextColor(r: number, g?: number, b?: number): jsPDF;
    text(text: string, x: number, y: number, options?: any): jsPDF;
    line(x1: number, y1: number, x2: number, y2: number): jsPDF;
    rect(x: number, y: number, w: number, h: number, style?: string): jsPDF;
    circle(x: number, y: number, r: number, style?: string): jsPDF;
    setPage(pageNumber: number): jsPDF;
    addPage(format?: string | [number, number], orientation?: 'portrait' | 'landscape'): jsPDF;
    deletePage(pageNumber: number): jsPDF;
    getNumberOfPages(): number;
    getCurrentPageInfo(): any;
  }
}
