// ===========================
// SAS Syntax Highlighting JavaScript
// ===========================

// SAS Language Definition for Syntax Highlighting
const SASLanguage = {
    // SAS Keywords
    keywords: [
        'data', 'set', 'run', 'proc', 'quit', 'if', 'then', 'else', 'do', 'end',
        'input', 'output', 'put', 'merge', 'by', 'retain', 'array', 'drop', 'keep',
        'where', 'when', 'select', 'otherwise', 'goto', 'link', 'return', 'stop',
        'abort', 'delete', 'rename', 'label', 'format', 'informat', 'length',
        'datalines', 'cards', 'infile', 'file', 'libname', 'filename', 'options',
        'title', 'footnote', 'ods', 'macro', 'mend', 'let', 'global', 'local'
    ],
    
    // SAS Procedures
    procedures: [
        'print', 'sort', 'means', 'freq', 'univariate', 'reg', 'glm', 'anova',
        'ttest', 'corr', 'report', 'tabulate', 'sql', 'transpose', 'contents',
        'datasets', 'format', 'import', 'export', 'gplot', 'gchart', 'sgplot'
    ],
    
    // SAS Functions
    functions: [
        'sum', 'mean', 'min', 'max', 'std', 'var', 'n', 'nmiss', 'abs', 'sqrt',
        'log', 'log10', 'exp', 'sin', 'cos', 'tan', 'int', 'round', 'ceil', 'floor',
        'substr', 'scan', 'trim', 'left', 'right', 'upcase', 'lowcase', 'propcase',
        'length', 'reverse', 'compress', 'index', 'find', 'translate', 'tranwrd',
        'cat', 'cats', 'catx', 'put', 'input', 'datepart', 'timepart', 'today',
        'mdy', 'ymd', 'year', 'month', 'day', 'weekday', 'intck', 'intnx', 'lag',
        'dif', 'ranuni', 'rannor', 'rand', 'missing', 'coalesce', 'ifn', 'ifc'
    ],
    
    // SAS Formats and Informats
    formats: [
        'best', 'comma', 'dollar', 'percent', 'date', 'datetime', 'time',
        'mmddyy', 'ddmmyy', 'yymmdd', 'worddate', 'weekdate', 'monname',
        'hex', 'binary', 'octal', 'char', 'ib', 'rb', 'pib', 'zd'
    ],
    
    // SAS Operators
    operators: [
        '=', '<', '>', '<=', '>=', '<>', '~=', '^=', 'eq', 'ne', 'lt', 'le',
        'gt', 'ge', 'in', 'not', 'and', 'or', '&', '|', '+', '-', '*', '/', '**'
    ]
};

// Initialize syntax highlighting when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    highlightAllSASCode();
    setupInteractiveFeatures();
});

// Main function to highlight all SAS code blocks
function highlightAllSASCode() {
    const codeBlocks = document.querySelectorAll('pre code.language-sas');
    
    codeBlocks.forEach(block => {
        highlightSASCode(block);
    });
}

// Highlight a single SAS code block
function highlightSASCode(element) {
    let code = element.textContent;
    
    // Preserve original code for copy functionality
    element.setAttribute('data-original-code', code);
    
    // Apply syntax highlighting
    code = highlightComments(code);
    code = highlightStrings(code);
    code = highlightNumbers(code);
    code = highlightKeywords(code);
    code = highlightProcedures(code);
    code = highlightFunctions(code);
    code = highlightFormats(code);
    code = highlightOperators(code);
    code = highlightMacroVariables(code);
    
    // Update element with highlighted code
    element.innerHTML = code;
}

// Highlight comments
function highlightComments(code) {
    // Multi-line comments /* ... */
    code = code.replace(/\/\*[\s\S]*?\*\//g, match => {
        return `<span class="comment">${escapeHtml(match)}</span>`;
    });
    
    // Single-line comments starting with *
    code = code.replace(/^\s*\*[^;]*;/gm, match => {
        return `<span class="comment">${escapeHtml(match)}</span>`;
    });
    
    return code;
}

// Highlight strings
function highlightStrings(code) {
    // Double-quoted strings
    code = code.replace(/"[^"]*"/g, match => {
        return `<span class="string">${escapeHtml(match)}</span>`;
    });
    
    // Single-quoted strings
    code = code.replace(/'[^']*'/g, match => {
        return `<span class="string">${escapeHtml(match)}</span>`;
    });
    
    return code;
}

// Highlight numbers
function highlightNumbers(code) {
    // Numbers (including decimals and scientific notation)
    code = code.replace(/\b\d+\.?\d*([eE][+-]?\d+)?\b/g, match => {
        return `<span class="number">${match}</span>`;
    });
    
    return code;
}

// Highlight keywords
function highlightKeywords(code) {
    const keywordRegex = new RegExp(`\\b(${SASLanguage.keywords.join('|')})\\b`, 'gi');
    
    code = code.replace(keywordRegex, (match, keyword) => {
        // Special highlighting for DATA, PROC, RUN, QUIT
        if (['data', 'proc', 'run', 'quit'].includes(keyword.toLowerCase())) {
            return `<span class="keyword-block">${match}</span>`;
        }
        return `<span class="keyword">${match}</span>`;
    });
    
    return code;
}

// Highlight procedures
function highlightProcedures(code) {
    // Match PROC followed by procedure name
    const procRegex = new RegExp(`(proc\\s+)(${SASLanguage.procedures.join('|')})\\b`, 'gi');
    
    code = code.replace(procRegex, (match, proc, procedure) => {
        return `<span class="keyword-block">${proc}</span><span class="function">${procedure}</span>`;
    });
    
    return code;
}

// Highlight functions
function highlightFunctions(code) {
    const functionRegex = new RegExp(`\\b(${SASLanguage.functions.join('|')})\\s*\\(`, 'gi');
    
    code = code.replace(functionRegex, (match, func) => {
        return `<span class="function">${func}</span>(`;
    });
    
    return code;
}

// Highlight formats
function highlightFormats(code) {
    // Match format specifications (e.g., dollar12.2, date9.)
    code = code.replace(/\b\w+\d*\.\d*\b/g, match => {
        // Check if it's a known format
        const baseFormat = match.replace(/\d+\.?\d*/, '');
        if (SASLanguage.formats.some(fmt => match.toLowerCase().startsWith(fmt))) {
            return `<span class="format">${match}</span>`;
        }
        return match;
    });
    
    return code;
}

// Highlight operators
function highlightOperators(code) {
    // Create regex pattern for operators (escape special characters)
    const escapedOperators = SASLanguage.operators.map(op => 
        op.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    );
    const operatorRegex = new RegExp(`(${escapedOperators.join('|')})`, 'g');
    
    code = code.replace(operatorRegex, match => {
        return `<span class="operator">${match}</span>`;
    });
    
    return code;
}

// Highlight macro variables
function highlightMacroVariables(code) {
    // Match &variable and &&variable patterns
    code = code.replace(/&+\w+\.?/g, match => {
        return `<span class="macro-var">${match}</span>`;
    });
    
    // Match %macro-function patterns
    code = code.replace(/%\w+/g, match => {
        return `<span class="macro-var">${match}</span>`;
    });
    
    return code;
}

// Setup interactive features
function setupInteractiveFeatures() {
    addLineNumbers();
    setupCodeTooltips();
    setupCodeExecution();
}

// Add line numbers to code blocks
function addLineNumbers() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        if (block.parentElement.classList.contains('line-numbers')) {
            const lines = block.innerHTML.split('\n');
            const lineNumbersDiv = document.createElement('div');
            lineNumbersDiv.className = 'line-numbers-rows';
            
            lines.forEach((line, index) => {
                const span = document.createElement('span');
                lineNumbersDiv.appendChild(span);
            });
            
            block.parentElement.appendChild(lineNumbersDiv);
        }
    });
}

// Setup tooltips for SAS keywords and functions
function setupCodeTooltips() {
    const tooltips = {
        'data': 'Creates or modifies a SAS dataset',
        'proc': 'Invokes a SAS procedure for data analysis',
        'set': 'Reads observations from one or more SAS datasets',
        'merge': 'Combines observations from two or more datasets',
        'by': 'Specifies variables for grouping or sorting',
        'retain': 'Retains variable values across iterations',
        'array': 'Defines an array of variables',
        'where': 'Filters observations based on conditions',
        'input': 'Describes how to read raw data',
        'format': 'Specifies how to display variable values'
    };
    
    // Add tooltips to keywords
    document.querySelectorAll('.keyword, .keyword-block').forEach(element => {
        const keyword = element.textContent.toLowerCase();
        if (tooltips[keyword]) {
            element.classList.add('code-tooltip');
            element.setAttribute('data-tooltip', tooltips[keyword]);
        }
    });
}

// Setup mock code execution (for demo purposes)
function setupCodeExecution() {
    const runButtons = document.querySelectorAll('.run-code-btn');
    
    runButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeBlock = this.closest('.code-example').querySelector('code');
            const code = codeBlock.getAttribute('data-original-code') || codeBlock.textContent;
            
            // Show loading state
            const outputDiv = document.getElementById('code-output');
            if (outputDiv) {
                outputDiv.innerHTML = '<div class="loading">Running SAS code...</div>';
                
                // Simulate execution delay
                setTimeout(() => {
                    outputDiv.innerHTML = generateMockOutput(code);
                }, 1000);
            }
        });
    });
}

// Generate mock output for demo
function generateMockOutput(code) {
    let output = '<div class="code-output-label">SAS Log</div>';
    output += '<pre class="code-output">';
    
    // Add timestamp
    const timestamp = new Date().toLocaleString();
    output += `NOTE: SAS session started at ${timestamp}\n\n`;
    
    // Analyze code and generate appropriate output
    if (code.includes('proc print')) {
        output += 'NOTE: There were 4 observations read from the data set WORK.EMPLOYEES.\n';
        output += 'NOTE: PROCEDURE PRINT used (Total process time):\n';
        output += '      real time           0.02 seconds\n';
        output += '      cpu time            0.01 seconds\n';
    } else if (code.includes('data')) {
        output += 'NOTE: The data set WORK.EMPLOYEES has 4 observations and 3 variables.\n';
        output += 'NOTE: DATA statement used (Total process time):\n';
        output += '      real time           0.01 seconds\n';
        output += '      cpu time            0.00 seconds\n';
    }
    
    output += '</pre>';
    return output;
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export functions for use in other scripts
window.SASHighlighter = {
    highlight: highlightSASCode,
    highlightAll: highlightAllSASCode
};