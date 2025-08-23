# Part 5: Data Input and Output - Mastering Data Movement

Welcome to Part 5 of our comprehensive SAS programming tutorial series! In this installment, we'll dive deep into one of the most essential skills for any SAS programmer: reading data from external sources and writing data to various formats. Whether you're importing customer records, processing survey responses, or exporting analysis results, mastering data input/output operations is crucial for effective data management.

## What You'll Learn

In this tutorial, you will:
- Read external data files using INFILE and INPUT statements
- Master various input techniques for different file formats
- Handle delimited, fixed-width, and complex file structures
- Write data to external files with FILE and PUT statements
- Use PROC IMPORT and EXPORT for quick data transfers
- Implement error handling and data validation strategies
- Optimize performance for large file operations

## Prerequisites

- Completion of Parts 1-4 of this tutorial series
- Understanding of SAS DATA steps and variable types
- Access to SAS (SAS OnDemand, SAS Studio, or any SAS installation)
- Basic knowledge of file paths and file systems
- Familiarity with common data formats (CSV, text files)

## Introduction to Data Input and Output

In the real world, data rarely exists solely within SAS. You'll frequently need to:
- Import data from CSV files, Excel spreadsheets, or text files
- Read fixed-width reports from legacy systems
- Export analysis results for use in other applications
- Create formatted reports for stakeholders

SAS provides multiple methods for handling these scenarios, from the flexible DATA step approaches to convenient PROC procedures. Understanding when and how to use each method will make you a more efficient programmer.

## Reading External Data with the DATA Step

The DATA step provides the most control and flexibility when reading external data. The key statements are:
- **INFILE**: Specifies the external file to read
- **INPUT**: Defines how to read the data

### Basic CSV File Reading

Let's start with a simple CSV file:

```sas
/* Sample CSV file: customers.csv
CustomerID,Name,Age,City,PurchaseAmount
1001,John Smith,35,New York,1250.50
1002,Jane Doe,28,Los Angeles,875.25
1003,Bob Johnson,42,Chicago,2100.00
*/

data customers;
    infile '/path/to/customers.csv' dsd firstobs=2;
    input CustomerID Name $ Age City $ PurchaseAmount;
run;

proc print data=customers noobs;
run;
```

Key points:
- `dsd` option handles comma-separated values and consecutive delimiters
- `firstobs=2` skips the header row
- Character variables are indicated with `$`

### Handling Different Delimiters

Not all files use commas as delimiters:

```sas
/* Tab-delimited file */
data sales_data;
    infile '/path/to/sales.txt' dlm='09'x firstobs=2;
    input Date :yymmdd10. Product $ Quantity Revenue;
    format Date date9.;
run;

/* Pipe-delimited file */
data inventory;
    infile '/path/to/inventory.dat' dlm='|' dsd;
    input ItemCode $ Description :$50. Stock Price;
run;
```

### Reading Fixed-Width Files

Legacy systems often produce fixed-width files where each field occupies specific columns:

```sas
/* Fixed-width file example:
Positions: 1-5: ID, 6-25: Name, 26-28: Age, 29-35: Salary
00123John Doe             025 50000
00124Jane Smith           032 65000
*/

data employees;
    infile '/path/to/employees.txt';
    input @1 ID $5.
          @6 Name $20.
          @26 Age 3.
          @29 Salary 7.;
run;
```

## Advanced INFILE Statement Options

The INFILE statement offers numerous options for handling various file scenarios:

### MISSOVER vs. TRUNCOVER

These options control how SAS handles short records:

```sas
/* MISSOVER: Sets remaining variables to missing */
data test_missover;
    infile datalines missover;
    input Name $ Score1 Score2 Score3;
    datalines;
John 85 90 95
Jane 88 92
Bob 90
;
run;

/* TRUNCOVER: Reads available data without going to next line */
data test_truncover;
    infile datalines truncover;
    input Name $10. Description $20.;
    datalines;
John       Senior Analyst
Jane       Manager
Bob        Jr Dev
;
run;
```

### Error Handling Options

Control how SAS responds to data errors:

```sas
data robust_import;
    infile '/path/to/messy_data.csv' dsd firstobs=2
           missover           /* Handle short records */
           lrecl=32767       /* Maximum record length */
           pad               /* Pad short records */
           termstr=crlf;     /* Handle different line endings */
    
    /* Use _ERROR_ and _N_ for debugging */
    input CustomerID Name :$50. Age Revenue;
    
    if _ERROR_ then do;
        put "Error on line " _N_ ": " _INFILE_;
        _ERROR_ = 0;  /* Reset error flag */
    end;
run;
```

## Advanced INPUT Techniques

The INPUT statement provides multiple ways to read data:

### Column Input with Pointers

Use pointer controls for precise positioning:

```sas
data complex_read;
    infile '/path/to/report.txt' firstobs=5;
    input @1 Date mmddyy10.
          @12 Region $3.
          @16 Sales comma12.2
          @30 Manager $20.
          @51 Status $1.;
    format Date date9. Sales dollar12.2;
run;
```

### Formatted Input

Read data with specific formats:

```sas
data formatted_data;
    infile datalines dsd;
    input TransDate :mmddyy10. 
          Time :time8.
          Amount :comma12.2
          Category :$15.
          Description :$50.;
    format TransDate date9. Time time8. Amount dollar12.2;
    datalines;
01/15/2024,14:30:00,"1,250.50",Electronics,Laptop purchase
02/20/2024,09:15:30,"350.25",Office,Printer supplies
;
run;
```

### Multiple Records Per Observation

Sometimes one observation spans multiple input lines:

```sas
data customer_addresses;
    infile datalines;
    input #1 CustomerID 4. Name $25.
          #2 Street $30.
          #3 City $20. State $2. Zip $5.;
    datalines;
1001 John Smith
123 Main Street
New York            NY10001
1002 Jane Doe
456 Oak Avenue
Los Angeles         CA90001
;
run;
```

### Named Input for Flexibility

Named input allows variables to appear in any order:

```sas
data flexible_input;
    infile datalines dsd;
    input Name=$ Age= City=$ Salary=;
    datalines;
Name=John,Age=30,City=Boston,Salary=75000
Age=25,Name=Jane,Salary=65000,City=Seattle
City=Denver,Salary=70000,Age=35,Name=Bob
;
run;
```

## Writing Data to External Files

The FILE and PUT statements allow you to write data:

### Basic File Output

```sas
data _null_;
    set customers;
    file '/path/to/output.csv' dsd;
    
    /* Write header on first observation */
    if _N_ = 1 then
        put 'CustomerID,Name,Age,City,PurchaseAmount';
    
    /* Write data */
    put CustomerID Name Age City PurchaseAmount;
run;
```

### Creating Formatted Reports

Generate professional-looking reports:

```sas
data _null_;
    set sales_summary end=last;
    file '/path/to/sales_report.txt' print header=header_section;
    
    /* Calculate running total */
    retain total 0;
    total + Revenue;
    
    /* Write detail lines */
    put @5 Region $10. @20 Product $15. @40 Revenue dollar12.2;
    
    /* Write total at end */
    if last then do;
        put @5 60*'-';
        put @5 'Grand Total:' @40 total dollar12.2;
    end;
    
    return;
    
    header_section:
        put @20 'Monthly Sales Report';
        put @20 19*'=';
        put;
        put @5 'Region' @20 'Product' @40 'Revenue';
        put @5 60*'-';
    return;
run;
```

### Dynamic File Names

Create multiple output files based on data:

```sas
data _null_;
    set sales_by_region;
    
    /* Create separate file for each region */
    length filename $50;
    filename = cats('/path/to/output/', Region, '_sales.csv');
    
    file dummy filevar=filename dsd mod;
    
    /* Write header if first record for region */
    if first.Region then
        put 'Date,Product,Quantity,Revenue';
    
    put Date Product Quantity Revenue;
run;
```

## PROC IMPORT: Quick and Easy Importing

PROC IMPORT provides a convenient way to import data without writing INPUT statements:

### Basic PROC IMPORT

```sas
/* Import CSV file */
proc import datafile='/path/to/sales_data.csv'
    out=sales_imported
    dbms=csv
    replace;
    getnames=yes;
    datarow=2;
run;

/* Import Excel file */
proc import datafile='/path/to/quarterly_report.xlsx'
    out=quarterly_data
    dbms=xlsx
    replace;
    sheet='Q1_2024';
    getnames=yes;
run;

/* Import tab-delimited file */
proc import datafile='/path/to/inventory.txt'
    out=inventory
    dbms=tab
    replace;
    getnames=yes;
run;
```

### PROC IMPORT Options

Control the import process:

```sas
proc import datafile='/path/to/customer_data.csv'
    out=customers
    dbms=csv
    replace;
    getnames=yes;      /* Use first row as variable names */
    datarow=2;         /* Data starts on row 2 */
    guessingrows=100;  /* Rows to scan for variable types */
run;

/* Review the generated DATA step code */
proc import datafile='/path/to/sample.csv'
    out=sample
    dbms=csv
    replace;
    getnames=yes;
run;

/* Check the log for the generated code */
```

## PROC EXPORT: Efficient Data Export

PROC EXPORT simplifies writing data to external files:

### Basic PROC EXPORT

```sas
/* Export to CSV */
proc export data=analysis_results
    outfile='/path/to/results.csv'
    dbms=csv
    replace;
    putnames=yes;
run;

/* Export to Excel */
proc export data=monthly_summary
    outfile='/path/to/summary.xlsx'
    dbms=xlsx
    replace;
    sheet='Summary';
run;

/* Export to tab-delimited */
proc export data=product_list
    outfile='/path/to/products.txt'
    dbms=tab
    replace;
run;
```

### Controlling Export Format

```sas
/* Export with specific delimiter */
proc export data=custom_output
    outfile='/path/to/output.txt'
    dbms=dlm
    replace;
    delimiter='|';
    putnames=no;
run;
```

## When to Use DATA Step vs. PROC IMPORT/EXPORT

### Use DATA Step When:
- You need precise control over data types
- The file has a complex structure
- You need to perform transformations during import
- You're dealing with fixed-width files
- Error handling is critical

### Use PROC IMPORT/EXPORT When:
- The file structure is straightforward
- You want SAS to determine variable types
- You're working with standard formats (CSV, Excel)
- You need a quick solution
- The file has consistent formatting

## Best Practices for Data Input/Output

### 1. Data Validation During Input

Always validate data as you read it:

```sas
data validated_import;
    infile '/path/to/customer_data.csv' dsd firstobs=2;
    input CustomerID Name :$50. Age Revenue;
    
    /* Validate age */
    if Age < 0 or Age > 120 then do;
        put "WARNING: Invalid age " Age " for customer " CustomerID;
        Age = .;
    end;
    
    /* Validate revenue */
    if Revenue < 0 then do;
        put "WARNING: Negative revenue " Revenue " for customer " CustomerID;
        delete;  /* Skip this record */
    end;
run;
```

### 2. Error Checking and Logging

Implement comprehensive error checking:

```sas
%let infile = /path/to/daily_transactions.csv;

data transactions 
     errors(keep=LineNumber ErrorDescription);
    
    infile "&infile" dsd firstobs=2 missover;
    length ErrorDescription $200;
    
    input TransID Date :mmddyy10. Amount Category $;
    
    /* Track line numbers */
    LineNumber = _N_;
    
    /* Check for missing required fields */
    if missing(TransID) then do;
        ErrorDescription = "Missing Transaction ID";
        output errors;
        delete;
    end;
    
    /* Validate date */
    if missing(Date) or Date > today() then do;
        ErrorDescription = cats("Invalid date: ", Date);
        output errors;
    end;
    else output transactions;
run;

/* Report errors */
proc print data=errors;
    title "Import Errors for &infile";
run;
```

### 3. Performance Considerations

Optimize large file operations:

```sas
/* Use PROC APPEND for large datasets */
proc append base=master_dataset
            data=daily_update;
run;

/* Use views for repetitive operations */
data monthly_import / view=monthly_import;
    infile '/path/to/monthly_*.csv' dsd firstobs=2;
    input Date :yymmdd10. Store $ Sales;
    format Date date9.;
run;

/* Buffer size optimization */
data large_file_import;
    infile '/path/to/huge_file.csv' dsd firstobs=2
           bufsize=32768  /* Increase buffer size */
           lrecl=4096;    /* Set appropriate record length */
    input /* variables */;
run;
```

## Practical Exercises

### Exercise 1: Multi-Format Import
Create a program that reads customer data from three different sources:
- A CSV file with basic customer information
- A fixed-width file with transaction history
- A pipe-delimited file with contact preferences

Merge these into a single comprehensive customer dataset.

### Exercise 2: Dynamic Report Generation
Write a program that:
1. Reads sales data from a CSV file
2. Creates separate output files for each region
3. Generates a summary report with totals
4. Exports the summary to Excel format

### Exercise 3: Error-Resistant Import
Develop a robust import process that:
- Handles missing values appropriately
- Validates data ranges
- Logs all errors to a separate dataset
- Creates a summary report of import statistics

### Exercise 4: Complex File Structure
Read a file where:
- The first line contains metadata
- Each customer record spans 3 lines
- Some fields are optional
- Date formats vary by record type

## Common Pitfalls and Solutions

### Problem 1: Truncated Character Variables
```sas
/* Problem: Names getting cut off */
data truncated;
    infile 'data.csv' dsd;
    input Name $ Age;  /* Default length is 8 */
run;

/* Solution: Specify length */
data fixed;
    infile 'data.csv' dsd;
    length Name $50;
    input Name $ Age;
run;
```

### Problem 2: Date Format Mismatches
```sas
/* Handle multiple date formats */
data flexible_dates;
    infile 'dates.csv' dsd;
    input DateText :$10. @@;
    
    /* Try multiple formats */
    if not missing(DateText) then do;
        Date = input(DateText, mmddyy10.);
        if missing(Date) then Date = input(DateText, date9.);
        if missing(Date) then Date = input(DateText, yymmdd10.);
    end;
    
    format Date date9.;
run;
```

### Problem 3: Embedded Delimiters
```sas
/* Handle commas within quoted fields */
data quoted_fields;
    infile 'complex.csv' dsd;  /* DSD handles quotes */
    input Company :$50. Revenue :comma12. Description :$100.;
run;
```

## Summary

In this tutorial, we've covered:
- Reading data using INFILE and INPUT statements
- Handling various file formats and delimiters
- Advanced INPUT techniques for complex data
- Writing data with FILE and PUT statements
- Using PROC IMPORT and EXPORT for convenience
- Best practices for robust data operations
- Performance optimization strategies

Mastering data input and output operations is fundamental to becoming an effective SAS programmer. These skills enable you to work with data from any source and share your results in any format required by your organization.

## What's Next?

In Part 6, we'll explore **Data Manipulation and Processing**, where you'll learn:
- Advanced data subsetting techniques
- Merging and joining datasets
- Data transformation strategies
- BY-group processing
- Creating complex derived variables

Continue practicing with different file formats and data structures. The more experience you gain with various data sources, the more prepared you'll be for real-world data challenges!

## Additional Resources

- [SAS Documentation: INFILE Statement](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/lestmtsglobal/p0deh6vbwdsxlgn13q5qzpgau5a5.htm)
- [SAS Documentation: INPUT Statement](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/lestmtsglobal/n0o71kgTvhmzvjn1teio37j0zovg.htm)
- [PROC IMPORT Guide](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/proc/n0bcvul5lfz6zon1mengm0r9fvjj.htm)
- Sample datasets for practice (available in course repository)

Happy coding, and see you in Part 6!