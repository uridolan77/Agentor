declare module 'react-to-print' {
  export type UseReactToPrintFn = () => void;
  export type UseReactToPrintHookContent = () => HTMLElement | null;

  export interface UseReactToPrintOptions {
    documentTitle?: string;
    onBeforeGetContent?: () => Promise<void> | void;
    onBeforePrint?: () => Promise<void> | void;
    onAfterPrint?: () => void;
    removeAfterPrint?: boolean;
    pageStyle?: string;
    printableElement?: HTMLElement;
  }

  export function useReactToPrint(options: {
    content: () => HTMLElement | null;
    documentTitle?: string;
    onBeforeGetContent?: () => Promise<void> | void;
    onBeforePrint?: () => Promise<void> | void;
    onAfterPrint?: () => void;
    removeAfterPrint?: boolean;
    pageStyle?: string;
    printableElement?: HTMLElement;
  }): UseReactToPrintFn;
}
