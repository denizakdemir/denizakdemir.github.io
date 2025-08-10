# Part 2: SAS Fundamentals - Building Your Foundation

Welcome to Part 2 of our comprehensive SAS programming tutorial series! Now that you've written your first SAS programs, it's time to build a solid foundation by understanding the core concepts that make SAS programming powerful and efficient.

## What You'll Learn

In this tutorial, you will:
- Master the fundamental difference between DATA steps and PROC steps
- Understand SAS libraries and how to manage datasets effectively
- Learn SAS syntax rules and naming conventions
- Explore the SAS log in detail for better debugging
- Write more complex programs with confidence
- Organize your code professionally

## Prerequisites

- Completion of Part 1: Getting Started with SAS
- Access to SAS (SAS OnDemand, SAS Studio, or any SAS installation)
- Basic understanding of running SAS programs

## SAS Program Structure: The Two Pillars

Every SAS program consists of two fundamental building blocks: DATA steps and PROC (procedure) steps. Understanding the distinction between these two is crucial for effective SAS programming.

### DATA Steps vs. PROC Steps

Think of SAS programming like cooking:
- **DATA steps** are like food preparation - cutting vegetables, marinating meat, mixing ingredients
- **PROC steps** are like cooking and presentation - grilling, plating, serving

Let's explore each in detail:

### DATA Steps: Creating and Manipulating Data

DATA steps are where you:
- Create new datasets
- Read raw data from external files
- Modify existing datasets
- Perform calculations and transformations
- Control which observations (rows) to include
- Create new variables (columns)

Here's the basic structure of a DATA step:

```sas
/* Basic DATA step structure */
data output_dataset;
    /* Instructions for creating/modifying data */
    /* These execute for each observation */
run;
```

**Detailed Explanation:**
- `data output_dataset;` - Begins the DATA step and names the dataset to create
- The statements between `data` and `run` execute once for each observation
- `run;` - Marks the end of the DATA step and triggers execution

### PROC Steps: Analyzing and Reporting Data

PROC steps are where you:
- Analyze data (statistics, frequencies, correlations)
- Create reports and visualizations
- Sort and manage datasets
- Export data to different formats
- Perform specialized analyses

Here's the basic structure of a PROC step:

```sas
/* Basic PROC step structure */
proc procedure_name data=input_dataset options;
    /* Additional statements specific to the procedure */
run;
```

**Detailed Explanation:**
- `proc procedure_name` - Invokes a specific SAS procedure
- `data=input_dataset` - Specifies which dataset to analyze
- `options` - Optional settings that modify procedure behavior
- `run;` - Executes the procedure

### Comparing DATA and PROC Steps

Let's see both in action with a complete example:

```sas
/* Step 1: DATA step - Create and prepare data */
data class_grades;
    input student_name $ exam1 exam2 final_exam;
    /* Calculate average grade for each student */
    average = (exam1 + exam2 + final_exam) / 3;
    
    /* Determine letter grade */
    if average >= 90 then grade = 'A';
    else if average >= 80 then grade = 'B';
    else if average >= 70 then grade = 'C';
    else if average >= 60 then grade = 'D';
    else grade = 'F';
    
    /* Determine pass/fail status */
    if grade in ('A', 'B', 'C') then status = 'Pass';
    else status = 'Fail';
    
    datalines;
Alice 85 92 88
Bob 78 75 82
Carol 95 98 96
David 65 70 68
Emma 88 85 90
;
run;

/* Step 2: PROC step - Analyze the data */
proc means data=class_grades;
    var exam1 exam2 final_exam average;
    title "Class Statistics Summary";
run;

/* Step 3: Another PROC step - Frequency analysis */
proc freq data=class_grades;
    tables grade status;
    title "Grade Distribution";
run;

/* Step 4: Another PROC step - Print the data */
proc print data=class_grades;
    title "Complete Class Grade Report";
    var student_name exam1 exam2 final_exam average grade status;
run;
```

**Key Observations:**
1. The DATA step creates the dataset and calculates new variables
2. Multiple PROC steps can analyze the same dataset
3. Each step ends with `run;`
4. DATA steps process data row by row
5. PROC steps process entire datasets

## Libraries and Datasets: Organizing Your Data

### Understanding SAS Libraries

A SAS library is like a folder or directory where SAS datasets are stored. Think of it as a filing cabinet:
- The library is the cabinet
- Datasets are the folders inside
- Variables are the papers in each folder

### Types of Libraries

#### 1. WORK Library (Temporary)
- Created automatically when SAS starts
- Deleted when SAS session ends
- Default location for datasets

```sas
/* Creating a dataset in WORK library (default) */
data employee_info;  /* Same as: data work.employee_info; */
    input emp_id name $ department $ salary;
    datalines;
101 John Sales 55000
102 Mary IT 65000
103 Bob HR 52000
;
run;
```

#### 2. Permanent Libraries
- Persist between SAS sessions
- Must be defined using LIBNAME statement
- Stored on your computer or server

```sas
/* Define a permanent library */
libname mydata '/home/username/sas_data';  /* Unix/Linux path */
libname mydata 'C:\Users\username\SAS_Data';  /* Windows path */

/* Create a dataset in permanent library */
data mydata.employee_info;
    input emp_id name $ department $ salary;
    datalines;
101 John Sales 55000
102 Mary IT 65000
103 Bob HR 52000
;
run;

/* Access the dataset later */
proc print data=mydata.employee_info;
    title "Permanent Employee Data";
run;
```

**Important Notes:**
- Library names (librefs) can be 1-8 characters
- Must start with a letter or underscore
- Can contain only letters, numbers, and underscores
- The physical path must exist before creating the library

### Working with Libraries

```sas
/* View all available libraries */
proc datasets;
    title "Available SAS Libraries";
run;

/* View contents of a specific library */
proc datasets library=work;
    title "Datasets in WORK Library";
run;

/* View detailed information about a dataset */
proc contents data=work.employee_info;
    title "Structure of Employee Info Dataset";
run;

/* Copy dataset between libraries */
data mydata.employee_backup;
    set work.employee_info;
run;

/* Delete a dataset */
proc datasets library=work;
    delete employee_info;
run;
```

### Two-Level Dataset Names

SAS uses two-level names to identify datasets:

```sas
/* Two-level naming: library.dataset */
data work.sales;      /* Temporary dataset in WORK */
data mydata.sales;    /* Permanent dataset in MYDATA */

/* One-level names default to WORK */
data sales;           /* Same as work.sales */
```

## Basic Syntax Rules: The Grammar of SAS

### 1. Statement Structure

Every SAS statement follows these rules:

```sas
/* Rule 1: Statements end with semicolons */
data newdata;
    set olddata;
    x = y + z;
run;

/* Rule 2: Statements can span multiple lines */
data employee_analysis;
    set employee_info;
    annual_bonus = salary * 0.10;
    total_compensation = salary + 
                        annual_bonus + 
                        benefits;
run;

/* Rule 3: Multiple statements can be on one line (not recommended) */
data test; x=1; y=2; z=x+y; run;  /* Works but hard to read */
```

### 2. Case Sensitivity Rules

Understanding what is and isn't case-sensitive in SAS:

```sas
/* SAS Keywords - NOT case sensitive */
DATA mydata;     /* Valid */
Data mydata;     /* Valid */
data MYDATA;     /* Valid */
DaTa MyDaTa;     /* Valid but silly */

/* Variable names - NOT case sensitive internally */
data test;
    Name = 'John';     /* Create variable */
    name = 'Jane';     /* Overwrites same variable */
    NAME = 'Bob';      /* Overwrites same variable */
run;

/* Variable values - CASE SENSITIVE */
data names;
    input name $ gender $;
    datalines;
John M
john m
JOHN M
;
run;

/* These are three different values! */
proc freq data=names;
    tables name;  /* Will show John, john, and JOHN separately */
run;
```

### 3. Naming Conventions

Rules for naming datasets, variables, and libraries:

```sas
/* Valid names must:
   - Start with letter or underscore
   - Contain only letters, numbers, underscores
   - Be 1-32 characters long (datasets/variables)
   - Be 1-8 characters long (libraries)
*/

/* Valid variable names */
data valid_names;
    employee_id = 101;        /* Good */
    _tempvar = 'temp';        /* Valid, starts with underscore */
    sales2024 = 50000;        /* Good */
    firstName = 'John';       /* Good - camelCase */
    first_name = 'John';      /* Good - snake_case */
run;

/* Invalid variable names - these will cause errors */
data invalid_names;
    2024sales = 50000;        /* ERROR: starts with number */
    employee-id = 101;        /* ERROR: contains hyphen */
    employee id = 101;        /* ERROR: contains space */
    employee.id = 101;        /* ERROR: contains period */
run;

/* Reserved words to avoid */
/* Don't use: data, proc, run, if, then, else, do, end, etc. */
```

### 4. Comments: Documenting Your Code

Two ways to add comments in SAS:

```sas
/* Method 1: Slash-asterisk comments
   - Can span multiple lines
   - Can be placed anywhere
   - Most common method
*/

data employee_analysis;
    set employees;
    /* Calculate annual bonus based on performance */
    if performance = 'Excellent' then
        bonus = salary * 0.15;  /* 15% for excellent */
    else if performance = 'Good' then
        bonus = salary * 0.10;  /* 10% for good */
    else
        bonus = salary * 0.05;  /* 5% for others */
run;

* Method 2: Asterisk comments;
* Must end with semicolon;
* Can only be used at statement boundaries;
* Good for temporarily disabling code;

* proc print data=test; run;  * This won't execute;
```

### 5. SAS System Options

Control how SAS behaves:

```sas
/* View current option settings */
proc options;
run;

/* Common options to set */
options linesize=80;        /* Characters per line in output */
options pagesize=60;        /* Lines per page in output */
options nodate;            /* Suppress date in output */
options nonumber;          /* Suppress page numbers */
options nocenter;          /* Left-align output */

/* Development options */
options mprint;            /* Show macro expansions */
options source;            /* Show source code in log */
options notes;             /* Show notes in log */
options errors=20;         /* Max errors before stopping */

/* Reset to defaults */
options date number center;
```

## Understanding the SAS Log: Your Debugging Companion

The SAS log is your primary tool for understanding what SAS is doing with your code. Let's explore it in detail.

### Components of the Log

```sas
/* Example program to generate various log messages */
data employee_analysis;
    input name $ age salary;
    
    /* This will generate a NOTE about numeric to character conversion */
    if age = '30' then age_group = 'Thirty';
    
    /* This will work correctly */
    if age = 30 then exact_thirty = 'Yes';
    
    datalines;
John 28 50000
Jane 30 55000
Bob 35 60000
Mary twentyfive 45000
;
run;
```

**Log Output Analysis:**

```
1    data employee_analysis;
2        input name $ age salary;
3        
4        /* This will generate a NOTE about numeric to character conversion */
5        if age = '30' then age_group = 'Thirty';
NOTE: Character values have been converted to numeric values at the places given by: (Line):(Column).
      5:14
      
6        
7        /* This will work correctly */
8        if age = 30 then exact_thirty = 'Yes';
9        
10       datalines;

NOTE: Invalid data for age in line 14 1-10.
RULE:     ----+----1----+----2----+----3----+----4
14        Mary twentyfive 45000
name=Mary age=. salary=45000 age_group=  exact_thirty=  _ERROR_=1 _N_=4
NOTE: The data set WORK.EMPLOYEE_ANALYSIS has 4 observations and 5 variables.
NOTE: DATA statement used (Total process time):
      real time           0.01 seconds
      cpu time            0.01 seconds
```

### Reading Log Messages Effectively

#### 1. Line Numbers
- Shows exactly where in your code something happened
- Essential for debugging

#### 2. Types of Messages

```sas
/* Generate different types of messages */

/* NOTE - Informational */
data test1;
    x = 1;
run;
/* Log: NOTE: The data set WORK.TEST1 has 1 observations and 1 variables. */

/* WARNING - Something unusual but not fatal */
data test2;
    input x y;
    datalines;
1 2 3
;
run;
/* Log: WARNING: Data line had more values than INPUT statement specified. */

/* ERROR - Must be fixed */
data test3;
    set nonexistent;
run;
/* Log: ERROR: File WORK.NONEXISTENT.DATA does not exist. */
```

### Performance Information

```sas
/* Create a larger dataset to see performance stats */
data large_data;
    do i = 1 to 1000000;
        x = ranuni(12345);
        y = rannor(12345);
        z = x * y;
        output;
    end;
run;

/* Log shows:
NOTE: The data set WORK.LARGE_DATA has 1000000 observations and 4 variables.
NOTE: DATA statement used (Total process time):
      real time           0.28 seconds
      cpu time            0.27 seconds
*/
```

## Your First Real Program: Putting It All Together

Let's create a comprehensive program that demonstrates all the concepts we've learned:

```sas
/***************************************************
* Program: Employee Analysis                       *
* Purpose: Demonstrate SAS fundamentals           *
* Author: Your Name                               *
* Date: Today's Date                              *
***************************************************/

/* Set up environment */
options nodate nonumber;
title; footnote;  /* Clear any existing titles */

/* Create a permanent library (adjust path as needed) */
libname mylib '/folders/myfolders/sas_data';

/* Step 1: Create employee dataset */
data work.employees;
    /* Define input variables and their types */
    input emp_id 
          first_name $ 
          last_name $ 
          department $ 
          hire_date :mmddyy10. 
          salary;
    
    /* Calculate derived variables */
    length full_name $50;
    full_name = catx(' ', first_name, last_name);
    
    /* Years of service calculation */
    years_service = intck('year', hire_date, today());
    
    /* Salary categorization */
    if salary < 40000 then salary_category = 'Entry';
    else if salary < 60000 then salary_category = 'Mid';
    else if salary < 80000 then salary_category = 'Senior';
    else salary_category = 'Executive';
    
    /* Format specifications */
    format hire_date mmddyy10.
           salary dollar12.0;
    
    /* Label variables for clarity */
    label emp_id = 'Employee ID'
          full_name = 'Full Name'
          department = 'Department'
          hire_date = 'Hire Date'
          salary = 'Annual Salary'
          years_service = 'Years of Service'
          salary_category = 'Salary Category';
    
    datalines;
1001 John Smith Sales 01/15/2018 55000
1002 Jane Doe IT 06/20/2019 68000
1003 Bob Johnson HR 03/10/2017 52000
1004 Mary Williams Finance 09/05/2020 61000
1005 David Brown IT 11/30/2016 75000
1006 Lisa Garcia Sales 04/22/2021 48000
1007 Mike Davis Finance 07/18/2019 69000
1008 Sarah Miller HR 02/14/2018 54000
;
run;

/* Step 2: Analyze the data */

/* Check dataset contents */
proc contents data=work.employees;
    title 'Employee Dataset Structure';
run;

/* Summary statistics by department */
proc means data=work.employees n mean min max;
    class department;
    var salary years_service;
    title 'Salary and Service Statistics by Department';
run;

/* Frequency analysis */
proc freq data=work.employees;
    tables department*salary_category / nocol nopercent;
    title 'Salary Categories by Department';
run;

/* Step 3: Create a report */
proc sort data=work.employees;
    by department descending salary;
run;

proc print data=work.employees label;
    by department;
    id emp_id;
    var full_name hire_date years_service salary salary_category;
    sum salary;
    title 'Employee Report by Department';
run;

/* Step 4: Save important results */
data mylib.employee_summary;
    set work.employees;
    /* Only keep employees with 3+ years service */
    where years_service >= 3;
    keep emp_id full_name department salary years_service;
run;

/* Verify the save */
proc print data=mylib.employee_summary;
    title 'Experienced Employees (3+ Years)';
run;

/* Clean up */
proc datasets library=work nolist;
    delete employees;
quit;
```

## Practical Exercises

### Exercise 1: Library Management

Create a program that:
1. Defines a permanent library
2. Creates a dataset in that library
3. Lists all datasets in the library
4. Copies a dataset from WORK to your permanent library

```sas
/* Your task: Complete this program */

/* 1. Define a library called 'mydata' */
libname _______ '/your/path/here';

/* 2. Create a dataset in the library */
data _______.products;
    input product_id product_name $ price quantity;
    datalines;
101 Laptop 999.99 50
102 Mouse 29.99 200
103 Keyboard 79.99 150
;
run;

/* 3. List datasets in the library */
proc _______ library=_______;
run;

/* 4. Create a temporary copy and then save it back */
data work.temp_products;
    set _______.products;
    total_value = price * quantity;
run;

data _______.products_with_value;
    set work.temp_products;
run;
```

**Solution:**
```sas
/* 1. Define a library called 'mydata' */
libname mydata '/folders/myfolders/data';

/* 2. Create a dataset in the library */
data mydata.products;
    input product_id product_name $ price quantity;
    datalines;
101 Laptop 999.99 50
102 Mouse 29.99 200
103 Keyboard 79.99 150
;
run;

/* 3. List datasets in the library */
proc datasets library=mydata;
run;

/* 4. Create a temporary copy and then save it back */
data work.temp_products;
    set mydata.products;
    total_value = price * quantity;
run;

data mydata.products_with_value;
    set work.temp_products;
run;
```

### Exercise 2: Understanding DATA vs PROC

Identify whether each task should use a DATA step or PROC step:

1. Calculate the average salary by department
2. Create a new variable for employee age
3. Sort data by hire date
4. Combine two datasets
5. Generate a frequency table
6. Filter data to include only specific departments
7. Create a statistical summary report
8. Add labels to variables

**Answers:**
1. PROC step (PROC MEANS)
2. DATA step
3. PROC step (PROC SORT)
4. DATA step
5. PROC step (PROC FREQ)
6. DATA step (though PROC SQL could also work)
7. PROC step
8. DATA step (though PROC DATASETS could also work)

### Exercise 3: Debugging Practice

Find and fix the errors in this program:

```sas
/* Program with intentional errors */
Data employee-data
    input name $ age salary;
    if age >= 30 then Group = Senior;
    else Group = 'Junior';
    datalines;
John 25 50000
Jane 35 65000
;
run

proc print data=employee-data;
    title 'Employee Report'
run;
```

**Solution:**
```sas
/* Fixed program */
data employee_data;  /* Fixed: no hyphens in dataset names */
    input name $ age salary;
    if age >= 30 then Group = 'Senior';  /* Fixed: quotes needed */
    else Group = 'Junior';
    datalines;
John 25 50000
Jane 35 65000
;
run;  /* Fixed: added semicolon */

proc print data=employee_data;  /* Fixed: dataset name */
    title 'Employee Report';  /* Fixed: added semicolon */
run;
```

## Best Practices for Code Organization

### 1. Use Consistent Naming Conventions

```sas
/* Good naming examples */
data customer_analysis;
    customer_id = 1001;
    first_purchase_date = '01JAN2020'd;
    total_purchases_2024 = 150;
    is_premium_member = 1;
run;

/* Avoid confusing names */
data data1;  /* Too generic */
    x = 1001;  /* What is x? */
    dt = '01JAN2020'd;  /* Unclear abbreviation */
run;
```

### 2. Structure Your Programs Logically

```sas
/*** 1. Setup ***/
options nodate nonumber;
libname project '/path/to/project';

/*** 2. Data Import ***/
/* Import raw data files */

/*** 3. Data Cleaning ***/
/* Fix data quality issues */

/*** 4. Data Transformation ***/
/* Create derived variables */

/*** 5. Analysis ***/
/* Statistical procedures */

/*** 6. Reporting ***/
/* Create final outputs */

/*** 7. Cleanup ***/
/* Remove temporary datasets */
```

### 3. Comment Strategically

```sas
/* Good commenting */
data sales_summary;
    set sales_detail;
    
    /* Calculate Q4 performance bonus:
       - 5% if sales > 100K
       - 3% if sales > 50K
       - 0% otherwise
    */
    if q4_sales > 100000 then q4_bonus = q4_sales * 0.05;
    else if q4_sales > 50000 then q4_bonus = q4_sales * 0.03;
    else q4_bonus = 0;
run;

/* Avoid obvious comments */
data test;
    x = 1;  /* Set x to 1 - obvious! */
run;
```

### 4. Use Meaningful Titles

```sas
/* Good titles provide context */
proc print data=employees;
    where department = 'IT' and salary > 60000;
    title 'IT Department Employees Earning Over $60,000';
    title2 'As of December 2024';
run;
```

## Common Pitfalls and How to Avoid Them

### 1. Forgetting RUN Statements

```sas
/* Problem: Missing RUN */
data test;
    x = 1;
    
proc print data=test;  /* ERROR: DATA step not terminated */
run;

/* Solution: Always include RUN */
data test;
    x = 1;
run;  /* Don't forget this! */

proc print data=test;
run;
```

### 2. Library Path Issues

```sas
/* Problem: Path doesn't exist */
libname mylib 'C:\NonexistentFolder';  /* ERROR */

/* Solution: Create folder first or use existing path */
/* In SAS OnDemand, use: */
libname mylib '/folders/myfolders/newfolder';
```

### 3. Dataset Name Confusion

```sas
/* Problem: Forgetting library reference */
data employees;
    /* ... */
run;

/* Later in program... */
proc print data=mylib.employees;  /* ERROR: Looking in wrong library */
run;

/* Solution: Be consistent with library usage */
proc print data=work.employees;  /* Correct library */
run;
```

## Summary and Key Takeaways

Congratulations! You now understand the fundamental building blocks of SAS programming:

### 1. **Program Structure**
- DATA steps create and modify data
- PROC steps analyze and report on data
- Every statement ends with a semicolon
- Programs flow from top to bottom

### 2. **Libraries and Datasets**
- WORK library for temporary data
- LIBNAME creates permanent libraries
- Two-level naming: library.dataset
- PROC DATASETS manages your data

### 3. **Syntax Rules**
- SAS keywords are not case-sensitive
- Variable values ARE case-sensitive
- Names must start with letter or underscore
- Comments help document your code

### 4. **The SAS Log**
- Your primary debugging tool
- Shows NOTEs, WARNINGs, and ERRORs
- Includes performance statistics
- Always check after running code

### 5. **Best Practices**
- Use meaningful variable names
- Comment your code strategically
- Organize programs logically
- Check the log after every submission

## What's Next?

In Part 3 of our series, "Mastering the DATA Step," we'll dive deep into:
- The Program Data Vector (PDV)
- How SAS processes data line by line
- Advanced INPUT techniques
- Conditional processing with IF-THEN-ELSE
- Creating multiple datasets in one DATA step
- Using RETAIN and sum statements

## Additional Resources

1. **Practice Dataset**: Download [employees.csv](/assets/data/sas-tutorial/employees.csv) to practice importing data
2. **SAS Documentation**: [DATA Step Documentation](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/lestmtsref/n1nh4bzuh8awn7n1h5abwgvh23xs.htm)
3. **Quick Reference**: [SAS Syntax Guide](/assets/data/sas-tutorial/sas-syntax-quick-reference.pdf)

Remember: The key to mastering SAS is practice. Try modifying the examples, experiment with different options, and always check your log!

Happy SAS programming! Keep building on these fundamentals, and you'll be writing complex SAS programs in no time.