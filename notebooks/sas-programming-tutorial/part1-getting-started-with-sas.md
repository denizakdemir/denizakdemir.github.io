---
title: "Getting Started with SAS - Your Gateway to Data Analytics"
author: "Deniz Akdemir"
date: 2025-08-09
categories: [SAS Programming, Tutorial]
tags: [SAS, Beginner, Data Analytics, Programming, SAS OnDemand, Getting Started]
render_with_liquid: false
toc: true
---

# Getting Started with SAS: Your Gateway to Data Analytics

Welcome to the first part of our comprehensive SAS programming tutorial series! Whether you're a complete beginner or someone looking to add SAS to your data analytics toolkit, this guide will help you take your first steps into the world of SAS programming.

## What You'll Learn

In this tutorial, you will:
- Understand what SAS is and why it's valuable in today's data-driven world
- Learn about career opportunities and industry demand for SAS skills
- Get hands-on experience setting up SAS for free
- Write and run your first SAS program
- Understand the basic structure of SAS programs
- Learn how to read and interpret SAS logs

## Prerequisites

None! This tutorial is designed for complete beginners. All you need is:
- A computer with internet access
- Basic computer skills (using a web browser, creating accounts)
- Enthusiasm to learn a powerful data analytics tool

## Why Learn SAS?

### Industry Demand and Career Opportunities

SAS (Statistical Analysis System) remains one of the most sought-after skills in data analytics, particularly in:

- **Healthcare and Pharmaceuticals**: Clinical trials, drug safety analysis, epidemiological studies
- **Banking and Finance**: Risk management, fraud detection, regulatory compliance
- **Government**: Census data analysis, economic forecasting, policy evaluation
- **Insurance**: Actuarial analysis, claims processing, customer segmentation
- **Retail**: Customer analytics, inventory optimization, market basket analysis

### The Numbers Speak for Themselves

According to recent job market data:
- Average SAS Programmer salary in the US: $75,000 - $120,000
- Over 70,000 job postings mention SAS skills annually
- 90% of Fortune 500 companies use SAS
- SAS certified professionals earn 15-25% more than non-certified peers

### SAS vs. Other Statistical Software

Let's compare SAS with other popular data analysis tools:

| Feature | SAS | R | Python | SPSS |
|---------|-----|---|---------|------|
| **Industry Adoption** | Very High (Enterprise) | High (Academia/Tech) | Very High (Tech) | Moderate |
| **Learning Curve** | Moderate | Steep | Moderate | Easy |
| **Cost** | Expensive (Free options available) | Free | Free | Expensive |
| **Support** | Professional | Community | Community | Professional |
| **Data Handling** | Excellent for large data | Good | Good | Limited |
| **Statistical Procedures** | Comprehensive | Comprehensive | Growing | Good |
| **Regulatory Compliance** | FDA approved | Limited | Limited | Limited |

### Why SAS Still Matters in 2025

1. **Reliability**: SAS has been tested and trusted for over 40 years
2. **Comprehensive Documentation**: Every procedure is thoroughly documented
3. **Audit Trail**: Built-in features for regulatory compliance
4. **Enterprise Support**: Professional support for mission-critical applications
5. **Backward Compatibility**: Code written decades ago still runs today

## How to Access SAS

The great news is that you can start learning SAS for free! Here are your options:

### 1. SAS OnDemand for Academics (Recommended for Beginners)

This is the easiest way to get started with SAS - completely free and runs in your web browser.

**Steps to Get Started:**

1. **Visit the SAS OnDemand Website**
   - Go to [https://welcome.oda.sas.com](https://welcome.oda.sas.com)
   - Click on "Register for an account"

2. **Create Your Profile**
   - Fill in your personal information
   - Use a valid email address (you'll need to verify it)
   - Choose "Independent Learner" if you're not affiliated with an institution

3. **Verify Your Email**
   - Check your inbox for a verification email from SAS
   - Click the verification link

4. **Complete Registration**
   - Set up your password (make it strong!)
   - Accept the terms and conditions
   - Complete the registration process

5. **Access SAS Studio**
   - Log in with your credentials
   - You'll see the SAS Studio interface load in your browser

### 2. SAS University Edition (Being Phased Out)

Note: SAS University Edition is being discontinued. New users should use SAS OnDemand instead.

### 3. Enterprise/Workplace Installations

If your organization has SAS:
- Contact your IT department for access
- You might have SAS installed locally or access via Citrix/Remote Desktop
- Enterprise versions often include additional modules

### 4. Cloud Options - SAS Viya

For advanced users and organizations:
- Cloud-native analytics platform
- Supports both SAS and open-source languages
- Requires enterprise licensing

## First Look at SAS Interface

Once you've logged into SAS OnDemand, you'll see SAS Studio - a modern, web-based interface for SAS programming.

### Understanding the SAS Studio Workspace

The interface consists of several key areas:

1. **Navigation Pane (Left Side)**
   - **Server Files and Folders**: Your file system
   - **Tasks and Utilities**: Point-and-click tools
   - **Snippets**: Code templates
   - **Libraries**: Data storage locations

2. **Work Area (Center)**
   - **Code Editor**: Where you write SAS programs
   - **Multiple tabs**: Work on several programs simultaneously
   - **Syntax highlighting**: Different colors for different code elements

3. **Results/Output Area**
   - **Results**: Shows output from procedures
   - **Log**: Displays messages about program execution
   - **Output Data**: View created datasets

4. **Toolbar**
   - **Run button**: Execute your code
   - **Save**: Save your programs
   - **Options**: Customize your environment

### Basic Navigation Tips

Let me show you some essential navigation commands:

```sas
/* This is a SAS comment - it won't execute but helps document your code */

* This is another way to write a comment in SAS;

/* To run code in SAS Studio:
   1. Type or paste your code in the editor
   2. Highlight the code you want to run (or run all)
   3. Click the "Run" button or press F3
*/
```

## Your First SAS Program

Let's write the traditional "Hello World" program in SAS!

### Example 1: Basic Output

```sas
/* My First SAS Program
   Author: Your Name
   Date: Today's Date
   Purpose: Display a message to the log
*/

data _null_;
    put "Hello, World! Welcome to SAS Programming!";
run;
```

**Detailed Explanation:**

1. **`data _null_;`**
   - `data` - This keyword starts a DATA step (we'll learn more about this later)
   - `_null_` - A special dataset name that means "don't create a dataset"
   - `;` - Every SAS statement ends with a semicolon (VERY IMPORTANT!)

2. **`put "Hello, World! Welcome to SAS Programming!";`**
   - `put` - Command to write text to the log
   - The text in quotes is what will be displayed
   - Again, note the semicolon at the end

3. **`run;`**
   - `run` - Tells SAS to execute the preceding DATA step
   - Without RUN, SAS waits for more statements

### Example 2: Creating Your First Dataset

Now let's create actual data:

```sas
/* Creating a simple dataset with employee information */

data employees;
    input name $ age salary;
    datalines;
John 35 55000
Sarah 28 48000
Mike 42 72000
Lisa 31 58000
;
run;

/* Display the data we just created */
proc print data=employees;
    title "My First Employee Dataset";
run;
```

**Detailed Explanation:**

1. **`data employees;`**
   - Creates a dataset named "employees"
   - This dataset will be stored in the WORK library (temporary storage)

2. **`input name $ age salary;`**
   - `input` - Tells SAS what variables to expect
   - `name $` - A character variable (the $ indicates character/text)
   - `age` - A numeric variable
   - `salary` - Another numeric variable

3. **`datalines;`**
   - Indicates that data values follow
   - Alternative keyword: `cards;` (they're synonymous)

4. **The actual data**
   - Each line represents one observation (row)
   - Values are separated by spaces
   - The order matches the INPUT statement

5. **`;`**
   - A semicolon on its own line ends the data input

6. **`proc print data=employees;`**
   - `proc` - Starts a PROCEDURE step
   - `print` - The procedure name (displays data)
   - `data=employees` - Specifies which dataset to print

7. **`title "My First Employee Dataset";`**
   - Adds a title to the output
   - Titles appear at the top of each page of output

### Example 3: Simple Calculations

Let's do some basic calculations:

```sas
/* Calculate annual bonus as 10% of salary */

data employees_with_bonus;
    set employees;  /* Read from existing dataset */
    bonus = salary * 0.10;
    total_compensation = salary + bonus;
    
    /* Add a label to make output clearer */
    label salary = "Annual Salary"
          bonus = "Annual Bonus (10%)"
          total_compensation = "Total Annual Compensation";
run;

/* Display the results with formatting */
proc print data=employees_with_bonus label;
    title "Employee Compensation Report";
    format salary bonus total_compensation dollar12.2;
run;
```

**Detailed Explanation:**

1. **`set employees;`**
   - Reads data from the existing "employees" dataset
   - Each observation is processed one at a time

2. **`bonus = salary * 0.10;`**
   - Creates a new variable called "bonus"
   - Calculates as 10% of salary
   - SAS uses standard math operators: + - * /

3. **`format salary bonus total_compensation dollar12.2;`**
   - `format` - Controls how values are displayed
   - `dollar12.2` - Display as currency with 12 total characters, 2 decimal places
   - Example: $55,000.00

## Understanding the SAS Log

The log is your best friend when programming in SAS. It tells you:
- What SAS did with your code
- Any errors or warnings
- Performance information

### Reading Log Messages

Let's intentionally create some errors to understand log messages:

```sas
/* Example with an error - missing semicolon */
data test
    input x y;
    datalines;
1 2
3 4
;
run;
```

**Log Output:**
```
ERROR: No DATALINES, INFILE, MERGE, SET, UPDATE or MODIFY statement.
ERROR 180-322: Statement is not valid or it is used out of proper order.
```

**What went wrong?** Missing semicolon after `data test`

### Types of Log Messages

1. **NOTES (Blue text)**
   - Informational messages
   - Example: "NOTE: The data set WORK.TEST has 2 observations and 2 variables."

2. **WARNINGS (Green text)**
   - Something unusual but not fatal
   - Example: "WARNING: Variable X already exists on file WORK.TEST."

3. **ERRORS (Red text)**
   - Something is wrong and needs fixing
   - Example: "ERROR: Variable X not found."

### Best Practices for Reading Logs

1. **Always check the log after running code**
2. **Start from the first error** - Later errors might be cascading effects
3. **Look for the line number** - SAS tells you where the error occurred
4. **Read the full error message** - It often suggests solutions

## Common Beginner Mistakes and How to Avoid Them

### 1. Forgetting Semicolons

**Wrong:**
```sas
data mydata
    input x y
run
```

**Right:**
```sas
data mydata;
    input x y;
run;
```

### 2. Incorrect Variable Types

**Wrong:**
```sas
data test;
    input name age;  /* name should be character */
    datalines;
John 25
;
run;
```

**Right:**
```sas
data test;
    input name $ age;  /* $ indicates character variable */
    datalines;
John 25
;
run;
```

### 3. Missing RUN Statements

**Wrong:**
```sas
proc print data=mydata;
/* Forgot run; - procedure won't execute */

proc means data=mydata;
run;
```

**Right:**
```sas
proc print data=mydata;
run;

proc means data=mydata;
run;
```

### 4. Case Sensitivity Confusion

Remember:
- **SAS keywords are NOT case-sensitive**: `DATA`, `data`, `Data` all work
- **Variable values ARE case-sensitive**: "John" ≠ "JOHN"
- **Variable names are NOT case-sensitive**: `Name`, `name`, `NAME` refer to the same variable

## Practical Exercises

### Exercise 1: Create a Student Dataset

Create a dataset with student information including name, age, and test score:

```sas
/* Your task: Create a dataset called 'students' with at least 5 students */
/* Include: name (character), age (numeric), test_score (numeric) */
/* Then display the data */

/* Write your code here */
```

**Solution:**
```sas
data students;
    input name $ age test_score;
    datalines;
Emma 20 95
Liam 19 87
Olivia 21 92
Noah 20 88
Ava 19 96
;
run;

proc print data=students;
    title "Student Test Scores";
run;
```

### Exercise 2: Calculate Grade Categories

Extend the student dataset to include grade categories:

```sas
/* Add a grade category based on test scores:
   90-100: A
   80-89: B
   70-79: C
   Below 70: D
*/

/* Write your code here */
```

**Solution:**
```sas
data students_graded;
    set students;
    
    if test_score >= 90 then grade = 'A';
    else if test_score >= 80 then grade = 'B';
    else if test_score >= 70 then grade = 'C';
    else grade = 'D';
run;

proc print data=students_graded;
    title "Student Grades Report";
run;
```

### Exercise 3: Basic Statistics

Calculate the average test score:

```sas
/* Use PROC MEANS to find the average test score */
/* Write your code here */
```

**Solution:**
```sas
proc means data=students;
    var test_score;
    title "Test Score Statistics";
run;
```

## Tips for Success

### 1. Practice Regularly
- Try to code a little bit every day
- Start with simple programs and gradually increase complexity

### 2. Use Comments Liberally
```sas
/* Good commenting practice */
data analysis;
    /* Calculate customer lifetime value */
    set customers;
    
    /* Average purchase × frequency × retention period */
    lifetime_value = avg_purchase * purchase_freq * retention_years;
run;
```

### 3. Organize Your Code
```sas
/******************************************
* Program: customer_analysis.sas          *
* Author: Your Name                       *
* Date: 2025-08-09                       *
* Purpose: Analyze customer purchase data *
******************************************/

/* Step 1: Import data */
/* Step 2: Clean and prepare */
/* Step 3: Analysis */
/* Step 4: Report results */
```

### 4. Save Your Work Frequently
- SAS Studio auto-saves, but manual saves are still important
- Use meaningful file names: `project1_data_cleaning.sas`

### 5. Build a Code Library
- Save useful code snippets
- Create templates for common tasks
- Document what each snippet does

## Summary and Key Takeaways

Congratulations! You've taken your first steps into SAS programming. Here's what we covered:

1. **Why SAS Matters**
   - Industry standard in healthcare, finance, and government
   - Excellent career opportunities
   - Robust and reliable for enterprise use

2. **Getting Started**
   - SAS OnDemand for Academics - free and browser-based
   - SAS Studio interface - modern and user-friendly

3. **Basic Programming Concepts**
   - DATA steps create and manipulate data
   - PROC steps analyze and report on data
   - Every statement ends with a semicolon
   - The log is essential for debugging

4. **Your First Programs**
   - Created datasets using INPUT and DATALINES
   - Performed calculations and transformations
   - Displayed results with PROC PRINT

## What's Next?

In Part 2 of our series, "SAS Fundamentals - Building Your Foundation," we'll dive deeper into:
- Understanding DATA steps vs. PROC steps
- Working with libraries and permanent datasets
- SAS naming conventions and rules
- More complex data operations
- Introduction to SAS functions

## Additional Resources

1. **SAS Documentation**: [support.sas.com](https://support.sas.com)
2. **SAS Communities**: [communities.sas.com](https://communities.sas.com)
3. **Practice Datasets**: Available in SAS OnDemand's SASHELP library

## Practice Dataset for Download

Save this as `first_practice.csv`:
```csv
employee_id,first_name,last_name,department,hire_date,salary
1001,John,Smith,Sales,01JAN2020,55000
1002,Sarah,Johnson,Marketing,15MAR2019,48000
1003,Mike,Williams,IT,22JUN2018,72000
1004,Lisa,Brown,HR,08SEP2021,58000
1005,David,Jones,Finance,30NOV2017,65000
```

Try importing this file using:
```sas
proc import datafile='path/to/first_practice.csv'
    out=practice_data
    dbms=csv
    replace;
run;
```

Happy SAS programming! Remember, every expert was once a beginner. Keep practicing, stay curious, and don't be afraid to make mistakes - they're the best teachers!