export const defaultFormattingRules = {
  pageLayout: {
    pageSize: 'A4',
    orientation: 'portrait',
    margins: {
      top: '2.54cm',
      bottom: '2.54cm',
      left: '2.54cm',
      right: '2.54cm'
    }
  },
  fonts: {
    title: {
      family: 'Arial',
      size: '24pt',
      weight: 'bold',
      color: '#000000'
    },
    heading1: {
      family: 'Arial',
      size: '18pt',
      weight: 'bold',
      color: '#000000'
    },
    heading2: {
      family: 'Arial',
      size: '16pt',
      weight: 'bold',
      color: '#000000'
    },
    heading3: {
      family: 'Arial',
      size: '14pt',
      weight: 'bold',
      color: '#000000'
    },
    body: {
      family: 'Arial',
      size: '12pt',
      weight: 'normal',
      color: '#000000'
    }
  },
  spacing: {
    paragraph: '1.5',
    section: '2',
    chapter: '3'
  },
  numbering: {
    chapters: true,
    sections: true,
    figures: true,
    tables: true
  },
  captions: {
    figures: {
      position: 'below',
      format: 'Figure {number}: {caption}',
      font: {
        family: 'Arial',
        size: '10pt',
        style: 'italic'
      }
    },
    tables: {
      position: 'above',
      format: 'Table {number}: {caption}',
      font: {
        family: 'Arial',
        size: '10pt',
        style: 'italic'
      }
    }
  }
};

export const templateTypes = {
  academic: {
    name: 'Academic Research Paper',
    description: 'Format for academic research papers with sections for abstract, methodology, and references',
    sections: [
      {
        name: 'Title Page',
        required: true,
        elements: ['title', 'author', 'institution', 'date']
      },
      {
        name: 'Abstract',
        required: true,
        elements: ['abstract']
      },
      {
        name: 'Table of Contents',
        required: true,
        elements: ['toc']
      },
      {
        name: 'Introduction',
        required: true,
        elements: ['heading', 'paragraphs']
      },
      {
        name: 'Methodology',
        required: true,
        elements: ['heading', 'paragraphs', 'figures', 'tables']
      },
      {
        name: 'Results',
        required: true,
        elements: ['heading', 'paragraphs', 'figures', 'tables']
      },
      {
        name: 'Discussion',
        required: true,
        elements: ['heading', 'paragraphs']
      },
      {
        name: 'References',
        required: true,
        elements: ['references']
      }
    ],
    formatting: {
      ...defaultFormattingRules,
      pageLayout: {
        ...defaultFormattingRules.pageLayout,
        margins: {
          top: '2.54cm',
          bottom: '2.54cm',
          left: '3.17cm',
          right: '3.17cm'
        }
      }
    }
  },
  business: {
    name: 'Business Proposal',
    description: 'Professional business proposal template with executive summary and financial projections',
    sections: [
      {
        name: 'Cover Page',
        required: true,
        elements: ['title', 'company', 'date', 'logo']
      },
      {
        name: 'Executive Summary',
        required: true,
        elements: ['heading', 'paragraphs']
      },
      {
        name: 'Company Overview',
        required: true,
        elements: ['heading', 'paragraphs', 'logo']
      },
      {
        name: 'Market Analysis',
        required: true,
        elements: ['heading', 'paragraphs', 'figures', 'tables']
      },
      {
        name: 'Financial Projections',
        required: true,
        elements: ['heading', 'tables', 'figures']
      }
    ],
    formatting: {
      ...defaultFormattingRules,
      fonts: {
        ...defaultFormattingRules.fonts,
        title: {
          family: 'Calibri',
          size: '28pt',
          weight: 'bold',
          color: '#1a237e'
        }
      }
    }
  },
  technical: {
    name: 'Technical Documentation',
    description: 'Technical documentation template with code examples and API references',
    sections: [
      {
        name: 'Title Page',
        required: true,
        elements: ['title', 'version', 'date']
      },
      {
        name: 'Overview',
        required: true,
        elements: ['heading', 'paragraphs']
      },
      {
        name: 'Installation',
        required: true,
        elements: ['heading', 'code', 'paragraphs']
      },
      {
        name: 'Usage',
        required: true,
        elements: ['heading', 'code', 'paragraphs', 'figures']
      },
      {
        name: 'API Reference',
        required: true,
        elements: ['heading', 'code', 'tables']
      }
    ],
    formatting: {
      ...defaultFormattingRules,
      fonts: {
        ...defaultFormattingRules.fonts,
        code: {
          family: 'Consolas',
          size: '11pt',
          weight: 'normal',
          color: '#000000'
        }
      }
    }
  }
}; 