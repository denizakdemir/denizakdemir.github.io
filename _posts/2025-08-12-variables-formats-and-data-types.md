---
title: "Variables, Formats, and Data Types - Getting Data Right"
author: denizakdemir
date: 2025-08-12 09:00:00 -0500
categories: [SAS Programming, Tutorial]
tags: [SAS, Variables, Formats, Informats, Data Types, Dates, Missing Values, Intermediate]
pin: false
math: true
mermaid: true
render_with_liquid: false
---

Welcome to Part 4 of our comprehensive SAS programming tutorial series! Now that you've mastered the DATA step, it's time to dive deep into the details that make your data accurate, readable, and professional: variables, formats, and data types. Understanding these concepts is crucial for data integrity and effective reporting.

## What You'll Learn

In this tutorial, you will:
- Master SAS data types and understand numeric precision
- Distinguish between informats and formats and when to use each
- Create custom formats with PROC FORMAT
- Handle missing values effectively across different data types
- Work with dates and times like a pro
- Apply variable attributes for better documentation and control

## Prerequisites

- Completion of Parts 1-3 of this tutorial series
- Understanding of DATA steps and basic SAS programming
- Access to SAS (SAS OnDemand, SAS Studio, or any SAS installation)
- Familiarity with creating and manipulating datasets

## Understanding SAS Data Types

SAS may seem simple with just two data types, but mastering their nuances is essential for accurate data processing.

### The Two Fundamental Types

SAS has only two data types:
1. **Numeric** - For numbers, dates, times, and mathematical operations
2. **Character** - For text, codes, and non-mathematical data

Let's explore each in detail:

### Numeric Variables and Precision

Numeric variables in SAS use floating-point representation, which has important implications:

```sas
/* Understanding numeric precision */
data numeric_precision;
    /* Default numeric length is 8 bytes */
    default_num = 123.456789012345;
    
    /* Specify different lengths */
    length short_num 3 
           medium_num 5 
           long_num 8;
    
    short_num = 123.456789012345;
    medium_num = 123.456789012345;
    long_num = 123.456789012345;
    
    /* Precision test */
    precise_calc = 1/3;
    test_equality = (0.1 + 0.2 = 0.3);  /* May surprise you! */
    
    /* Large numbers */
    big_number = 999999999999999;
    bigger_number = 9999999999999999;
    
    put "Default: " default_num best32.;
    put "Short (3): " short_num best32.;
    put "Medium (5): " medium_num best32.;
    put "Long (8): " long_num best32.;
    put "1/3 = " precise_calc best32.;
    put "0.1 + 0.2 = 0.3? " test_equality;
run;
```

**Important Notes on Numeric Precision:**
- Default length is 8 bytes (maximum precision)
- Lengths 3-7 bytes save space but reduce precision
- Floating-point arithmetic can cause unexpected results
- Use ROUND function for comparisons when needed

### Safe Numeric Comparisons

```sas
/* Handling floating-point comparisons safely */
data safe_comparisons;
    a = 0.1;
    b = 0.2;
    c = 0.3;
    
    /* Unsafe comparison */
    exact_equal = (a + b = c);  /* Might be FALSE! */
    
    /* Safe comparison with rounding */
    safe_equal = (round(a + b, 0.0001) = round(c, 0.0001));
    
    /* Using a tolerance */
    tolerance = 0.0000001;
    within_tolerance = (abs((a + b) - c) < tolerance);
    
    put "Exact equality: " exact_equal;
    put "Safe equality: " safe_equal;
    put "Within tolerance: " within_tolerance;
run;
```

### Character Variables and Length

Character variables store text data with some important considerations:

```sas
/* Character variable length management */
data character_length;
    /* Default length is determined by first assignment */
    default_char = 'Hi';  /* Length becomes 2 */
    default_char = 'Hello World';  /* Truncated to 'He' */
    
    /* Explicit length declaration */
    length name $50 
           code $10 
           description $200;
    
    name = 'John Smith';
    code = 'PROD-2024';
    description = 'This is a detailed product description';
    
    /* Demonstrating truncation */
    length short_var $5;
    short_var = 'This will be truncated';  /* Becomes 'This ' */
    
    /* Leading/trailing spaces */
    padded = '  spaces  ';
    trimmed = trim(padded);
    stripped = strip(padded);  /* Removes both leading and trailing */
    
    put "Default char: '" default_char "'";
    put "Short var: '" short_var "'";
    put "Padded: '" padded "'";
    put "Trimmed: '" trimmed "'";
    put "Stripped: '" stripped "'";
run;
```

### Character Variable Best Practices

```sas
/* Best practices for character variables */
data char_best_practices;
    /* Always declare length before use */
    length customer_id $10 
           full_name $100 
           email $50 
           status_code $1;
    
    /* Efficient length allocation */
    customer_id = 'CUST00123';     /* 10 chars is sufficient */
    full_name = 'Smith, John Q.';   /* Allow for long names */
    email = 'john.smith@email.com'; /* Standard email length */
    status_code = 'A';              /* Single character codes */
    
    /* Concatenation with proper spacing */
    length first_name $30 last_name $30;
    first_name = 'John';
    last_name = 'Smith';
    
    /* Different concatenation methods */
    concat1 = trim(first_name) || ' ' || trim(last_name);
    concat2 = catx(' ', first_name, last_name);  /* Preferred */
    concat3 = cats(first_name, last_name);       /* No separator */
    
    put "Concat1: " concat1;
    put "Concat2: " concat2;
    put "Concat3: " concat3;
run;
```

## Informats vs. Formats: Reading vs. Displaying

Understanding the difference between informats and formats is crucial for proper data handling.

### Informats: Reading Data INTO SAS

Informats tell SAS how to read raw data:

```sas
/* Common informats for reading data */
data informat_examples;
    /* Reading different date formats */
    input @1 date1 mmddyy10.
          @12 date2 date9.
          @22 date3 yymmdd10.
          @33 date4 anydtdte10.;
    
    /* Reading numeric data with special characters */
    input @1 salary dollar10.
          @12 percent percent5.
          @18 comma_num comma12.2;
    
    /* Reading character data */
    input @1 name $20.
          @22 code $char10.  /* Preserves leading spaces */
          @33 hex_value $hex4.;
    
    datalines;
01/15/2024 15JAN2024 2024-01-15 15/01/2024
$75,500.00 85.5% 1,234,567.89
John Smith          ABC     41424344
;
run;

proc print data=informat_examples;
    format date1-date4 mmddyy10.
           salary dollar10.2
           percent percent8.1;
run;
```

### Common Informats Reference

```sas
/* Comprehensive informat examples */
data informat_reference;
    /* Numeric informats */
    input @1 standard_num 8.
          @10 comma_num comma10.
          @21 dollar_amt dollar10.
          @32 percent_val percent5.
          @38 scientific e10.;
    
    /* Date/time informats */
    input @1 mdy_date mmddyy10.
          @12 dmy_date ddmmyy10.
          @23 iso_date yymmdd10.
          @34 sas_date date9.
          @44 datetime datetime20.
          @65 time time8.;
    
    /* Character informats */
    input @1 fixed_char $10.     /* Fixed width */
          @12 vary_char $varying20. n  /* Variable width */
          @33 upcase $upcase10.  /* Convert to uppercase */
          @44 binary $binary8.;  /* Binary representation */
    
    datalines;
12345678 12,345.67 $9,999.99 45.5% 1.23E+05
01/15/2024 15/01/2024 2024-01-15 15JAN2024 15JAN2024:10:30:45 10:30:45
Fixed Text 10 Variable Text LOWERCASE  01000001
;
run;
```

### Formats: Displaying Data FROM SAS

Formats control how SAS displays values:

```sas
/* Common formats for displaying data */
data format_examples;
    /* Create sample data */
    num_value = 1234567.89;
    date_value = '15JAN2024'd;
    time_value = '10:30:45't;
    datetime_value = '15JAN2024:10:30:45'dt;
    decimal_value = 0.756;
    
    /* Display with different formats */
    put "Original number: " num_value;
    put "Comma format: " num_value comma12.2;
    put "Dollar format: " num_value dollar15.2;
    put "Scientific: " num_value e10.;
    put "Best format: " num_value best12.;
    
    put / "Date formats:";
    put "MMDDYY10.: " date_value mmddyy10.;
    put "DATE9.: " date_value date9.;
    put "WORDDATE.: " date_value worddate.;
    put "WEEKDATE.: " date_value weekdate.;
    
    put / "Decimal as percent: " decimal_value percent8.1;
run;
```

### Creating Custom Formats with PROC FORMAT

One of SAS's most powerful features is creating custom formats:

```sas
/* Creating custom formats */
proc format;
    /* Numeric format for age groups */
    value age_group
        low - 12 = 'Child'
        13 - 19 = 'Teenager'
        20 - 29 = 'Young Adult'
        30 - 49 = 'Adult'
        50 - 64 = 'Middle Age'
        65 - high = 'Senior';
    
    /* Character format for status codes */
    value $status_fmt
        'A' = 'Active'
        'I' = 'Inactive'
        'P' = 'Pending'
        'T' = 'Terminated'
        other = 'Unknown';
    
    /* Numeric format with specific ranges */
    value score_fmt
        0 - 59 = 'F - Failing'
        60 - 69 = 'D - Below Average'
        70 - 79 = 'C - Average'
        80 - 89 = 'B - Good'
        90 - 100 = 'A - Excellent'
        . = 'Not Graded'
        other = 'Invalid Score';
    
    /* Picture format for custom display */
    picture phone_fmt
        0000000000 - 9999999999 = '(999) 999-9999' 
        (prefix='Phone: ');
    
    /* Nested format using other formats */
    value risk_level
        low - 0.3 = 'Low Risk'
        0.3 <- 0.7 = 'Medium Risk'
        0.7 <- high = 'High Risk';
run;

/* Using custom formats */
data test_custom_formats;
    input age score status $ phone;
    format age age_group.
           score score_fmt.
           status $status_fmt.
           phone phone_fmt.;
    datalines;
8 95 A 5551234567
25 72 I 5559876543
45 88 P 5555551212
67 . T 5551112222
;
run;

proc print data=test_custom_formats;
    title "Data Displayed with Custom Formats";
run;
```

### Advanced Format Techniques

```sas
/* Advanced custom format techniques */
proc format;
    /* Multi-label format for overlapping categories */
    value age_multi (multilabel)
        0 - 17 = 'Minor'
        0 - 12 = 'Child'
        13 - 19 = 'Teen'
        18 - 24 = 'Young Adult'
        21 - 65 = 'Working Age'
        65 - high = 'Retirement Age';
    
    /* Conditional format based on other values */
    value income_cat
        low - 30000 = 'Low Income'
        30000 <- 60000 = 'Middle Income'
        60000 <- 100000 = 'Upper Middle'
        100000 <- high = 'High Income';
    
    /* Format with specific value handling */
    value special_missing
        . = 'Not Available'
        .A = 'Not Applicable'
        .B = 'Refused to Answer'
        .Z = 'Unknown'
        other = [8.2];  /* Use default numeric format */
run;

/* Using formats for data categorization */
data categorized_data;
    input customer_id income age response;
    
    /* Create categories using formats */
    income_category = put(income, income_cat.);
    age_category = put(age, age_group.);
    
    /* Format for display */
    format income dollar10.0
           response special_missing.;
    
    datalines;
101 25000 25 85.5
102 75000 45 .
103 150000 67 .A
104 45000 30 92.3
;
run;
```

## Variable Attributes Deep Dive

Variable attributes define how SAS stores and displays your data:

### Setting and Managing LENGTH

```sas
/* Length attribute management */
data length_management;
    /* Method 1: LENGTH statement (recommended) */
    length id $10 
           name $50 
           amount 8 
           flag 3;
    
    /* Method 2: ATTRIB statement (comprehensive) */
    attrib customer_id length=$15 label='Customer Identifier'
           balance length=8 format=dollar12.2 label='Account Balance'
           status length=$1 format=$status_fmt. label='Status Code';
    
    /* Length implications for numeric variables */
    length num3 3 num4 4 num5 5 num8 8;
    
    /* Test precision limits */
    num3 = 999999;      /* Max accurate value for length 3 */
    num4 = 9999999;     /* Max accurate value for length 4 */
    num5 = 999999999;   /* Max accurate value for length 5 */
    num8 = 999999999999999; /* Max accurate value for length 8 */
    
    /* Character length is straightforward */
    id = 'CUST123';
    name = 'Very Long Customer Name That Fits';
run;

/* View the impact of length */
proc contents data=length_management;
    title "Variable Attributes and Storage";
run;
```

### Labels for Documentation

Labels provide meaningful descriptions for variables:

```sas
/* Comprehensive labeling example */
data well_labeled_data;
    /* Using LABEL statement */
    label customer_id = 'Unique Customer Identifier'
          first_name = 'Customer First Name'
          last_name = 'Customer Last Name'
          birth_date = 'Date of Birth'
          account_open_date = 'Account Opening Date'
          account_balance = 'Current Account Balance'
          credit_score = 'FICO Credit Score'
          risk_category = 'Risk Assessment Category';
    
    /* Using ATTRIB for everything at once */
    attrib total_transactions length=8 
           format=comma10. 
           label='Total Number of Transactions'
           
           avg_transaction length=8 
           format=dollar10.2 
           label='Average Transaction Amount'
           
           last_activity length=8 
           format=mmddyy10. 
           label='Date of Last Account Activity';
    
    /* Input some sample data */
    input customer_id $ first_name $ last_name $ 
          birth_date :mmddyy10. credit_score;
    
    datalines;
C001 John Smith 03/15/1980 750
C002 Jane Doe 07/22/1975 680
;
run;

/* Labels appear in procedures */
proc print data=well_labeled_data label;
    title "Customer Data with Labels";
run;

proc contents data=well_labeled_data;
    title "Dataset Documentation";
run;
```

### Format Persistence and Inheritance

```sas
/* Understanding format persistence */
data format_persistence;
    /* Permanent format assignment */
    format sale_date mmddyy10.
           amount dollar10.2
           tax_rate percent8.2;
    
    sale_date = '15JAN2024'd;
    amount = 1250.50;
    tax_rate = 0.0875;
run;

/* Formats persist when creating new datasets */
data inherited_formats;
    set format_persistence;
    total_with_tax = amount * (1 + tax_rate);
    /* total_with_tax inherits dollar format automatically */
run;

/* Removing formats */
data remove_formats;
    set inherited_formats;
    
    /* Remove format from specific variable */
    format amount;  /* No format specified removes it */
    
    /* Remove all formats */
    format _all_;   /* Removes formats from all variables */
run;

/* Changing formats */
data change_formats;
    set format_persistence;
    
    /* Override inherited format */
    format sale_date date9.
           amount comma10.0
           tax_rate 8.4;
run;
```

## Working with Missing Values

Missing values require special attention in SAS:

### Types of Missing Values

```sas
/* Understanding missing values */
data missing_value_types;
    /* Numeric missing values */
    regular_missing = .;
    special_missing_a = .A;
    special_missing_b = .B;
    special_missing_z = .Z;
    
    /* Character missing values */
    char_missing = '';      /* Empty string */
    char_blank = ' ';       /* Single space */
    
    /* Missing values in calculations */
    x = 10;
    y = .;
    sum_xy = x + y;         /* Result is missing */
    sum_function = sum(x, y); /* Result is 10 - SUM ignores missing */
    
    /* Testing for missing values */
    if missing(y) then miss_flag1 = 'Y';
    if y = . then miss_flag2 = 'Y';
    if nmiss(x, y) > 0 then any_missing = 'Y';
    
    put "Regular sum: " sum_xy=;
    put "SUM function: " sum_function=;
run;
```

### Functions for Handling Missing Data

```sas
/* Missing value functions */
data missing_functions;
    input id q1 q2 q3 q4 q5;
    
    /* Count missing values */
    n_missing = nmiss(of q1-q5);
    n_complete = n(of q1-q5);
    
    /* Calculate with missing values */
    mean_all = mean(of q1-q5);  /* Ignores missing */
    sum_all = sum(of q1-q5);    /* Ignores missing */
    
    /* Replace missing values */
    array questions[5] q1-q5;
    array q_filled[5];
    
    do i = 1 to 5;
        /* Coalesce - use first non-missing */
        q_filled[i] = coalesce(questions[i], 0);
    end;
    
    /* Conditional processing with missing */
    if n_missing = 0 then completeness = 'Complete';
    else if n_missing <= 2 then completeness = 'Partial';
    else completeness = 'Insufficient';
    
    datalines;
101 90 85 . 88 92
102 . . 75 80 .
103 88 92 95 90 94
104 . . . . 85
;
run;

proc print data=missing_functions;
    title "Missing Value Analysis";
run;
```

### Best Practices for Missing Values

```sas
/* Missing value best practices */
data handle_missing_properly;
    input customer_id $ age income satisfaction_score;
    
    /* 1. Document missing value meanings */
    if missing(age) then age_status = 'Not Provided';
    else if age = .R then age_status = 'Refused';
    else if age = .D then age_status = 'Don''t Know';
    else age_status = 'Valid';
    
    /* 2. Imputation strategies */
    /* Simple mean imputation */
    if missing(income) then do;
        income_imputed = 50000;  /* Population mean */
        imputed_flag = 'Y';
    end;
    else do;
        income_imputed = income;
        imputed_flag = 'N';
    end;
    
    /* 3. Exclude or include in calculations */
    if not missing(satisfaction_score) then do;
        valid_response = 'Y';
        /* Include in analysis */
    end;
    
    /* 4. Create missing indicators */
    missing_age = missing(age);
    missing_income = missing(income);
    missing_satisfaction = missing(satisfaction_score);
    
    datalines;
C001 35 75000 8.5
C002 .  65000 9.0
C003 42 .     7.5
C004 .R 80000 .
C005 28 55000 .D
;
run;
```

## Date and Time Handling

SAS's date and time system is powerful but requires understanding:

### SAS Date System Explained

```sas
/* Understanding SAS dates */
data sas_date_system;
    /* SAS dates are numbers - days since Jan 1, 1960 */
    sas_epoch = 0;
    format sas_epoch date9.;
    put "SAS Epoch (0): " sas_epoch date9.;
    
    /* Positive and negative dates */
    before_epoch = -365;  /* Jan 1, 1959 */
    after_epoch = 365;    /* Jan 1, 1961 */
    format before_epoch after_epoch date9.;
    
    /* Date constants */
    date1 = '01JAN2024'd;
    date2 = "15MAR2024"d;  /* Single or double quotes */
    today_date = today();
    
    /* Common date creation */
    from_mdy = mdy(12, 25, 2024);  /* Christmas 2024 */
    from_string = input('2024-12-25', yymmdd10.);
    
    put "Before epoch: " before_epoch=;
    put "After epoch: " after_epoch=;
    put "Today: " today_date= date9.;
    put "Christmas: " from_mdy= worddate.;
run;
```

### Date Functions and Calculations

```sas
/* Comprehensive date functions */
data date_calculations;
    /* Sample dates */
    start_date = '01JAN2024'd;
    end_date = '31DEC2024'd;
    birth_date = '15MAR1990'd;
    
    /* Extract date components */
    year_part = year(start_date);
    month_part = month(start_date);
    day_part = day(start_date);
    quarter = qtr(start_date);
    week_number = week(start_date);
    weekday = weekday(start_date);  /* 1=Sunday, 2=Monday, etc. */
    
    /* Date arithmetic */
    days_between = end_date - start_date;
    age_days = today() - birth_date;
    age_years = int(age_days / 365.25);
    
    /* INTCK - count intervals */
    months_between = intck('month', start_date, end_date);
    years_between = intck('year', start_date, end_date);
    exact_years = intck('year', birth_date, today(), 'C'); /* Continuous */
    
    /* INTNX - increment dates */
    next_month = intnx('month', start_date, 1);
    prev_quarter = intnx('quarter', start_date, -1);
    year_start = intnx('year', start_date, 0, 'B');  /* Beginning */
    year_end = intnx('year', start_date, 0, 'E');    /* End */
    
    /* Business day calculations */
    next_weekday = intnx('weekday', start_date, 1);
    
    format start_date end_date birth_date next_month 
           prev_quarter year_start year_end next_weekday date9.;
run;

proc print data=date_calculations;
    title "Date Calculations and Functions";
run;
```

### Working with Times and Datetimes

```sas
/* Time and datetime handling */
data time_datetime;
    /* Time values - seconds since midnight */
    morning = '08:30:00't;
    afternoon = '14:45:30't;
    current_time = time();
    
    /* Datetime values - seconds since Jan 1, 1960 midnight */
    meeting_dt = '15JAN2024:10:30:00'dt;
    now_dt = datetime();
    
    /* Extract components */
    hour_part = hour(morning);
    minute_part = minute(morning);
    second_part = second(morning);
    
    /* From datetime */
    date_from_dt = datepart(meeting_dt);
    time_from_dt = timepart(meeting_dt);
    
    /* Calculations */
    time_diff_seconds = afternoon - morning;
    time_diff_hours = time_diff_seconds / 3600;
    
    /* Creating datetime from date and time */
    some_date = '15MAR2024'd;
    some_time = '15:30:00't;
    combined_dt = dhms(some_date, 0, 0, some_time);
    
    /* Formatting */
    format morning afternoon current_time time8.
           meeting_dt now_dt datetime20.
           date_from_dt date9.
           time_from_dt time8.
           combined_dt datetime20.;
run;

proc print data=time_datetime;
    title "Time and Datetime Examples";
run;
```

### Date Format Gallery

```sas
/* Comprehensive date format examples */
data date_format_gallery;
    sample_date = '15MAR2024'd;
    sample_time = '14:30:45't;
    sample_datetime = '15MAR2024:14:30:45'dt;
    
    /* Display same date in multiple formats */
    array formats[15] $32 fmt1-fmt15;
    array descriptions[15] $32 desc1-desc15;
    
    formats[1] = put(sample_date, date9.);         desc1 = 'DATE9.';
    formats[2] = put(sample_date, date11.);        desc2 = 'DATE11.';
    formats[3] = put(sample_date, mmddyy10.);      desc3 = 'MMDDYY10.';
    formats[4] = put(sample_date, mmddyy8.);       desc4 = 'MMDDYY8.';
    formats[5] = put(sample_date, ddmmyy10.);      desc5 = 'DDMMYY10.';
    formats[6] = put(sample_date, yymmdd10.);      desc6 = 'YYMMDD10.';
    formats[7] = put(sample_date, worddate.);      desc7 = 'WORDDATE.';
    formats[8] = put(sample_date, weekdate.);      desc8 = 'WEEKDATE.';
    formats[9] = put(sample_date, monyy7.);        desc9 = 'MONYY7.';
    formats[10] = put(sample_date, year4.);        desc10 = 'YEAR4.';
    formats[11] = put(sample_date, qtr1.);         desc11 = 'QTR1.';
    formats[12] = put(sample_date, julian5.);      desc12 = 'JULIAN5.';
    formats[13] = put(sample_date, eurdfdd10.);    desc13 = 'EURDFDD10.';
    formats[14] = put(sample_time, time8.);        desc14 = 'TIME8.';
    formats[15] = put(sample_datetime, datetime20.); desc15 = 'DATETIME20.';
    
    /* Output each format */
    do i = 1 to 15;
        format_name = descriptions[i];
        formatted_value = formats[i];
        output;
    end;
    
    keep format_name formatted_value;
run;

proc print data=date_format_gallery;
    title "Date and Time Format Gallery";
run;
```

## Practical Examples: Real-World Applications

### Example 1: Customer Data Processing

```sas
/* Real-world customer data processing */
data raw_customer_input;
    input @1 cust_id $10.
          @12 join_date mmddyy10.
          @23 last_purchase mmddyy10.
          @34 total_purchases dollar12.
          @47 status $1.
          @49 phone $10.
          @60 email $50.;
    datalines;
CUST001234 01/15/2020 12/28/2023  $12,567.89 A 5551234567 john.smith@email.com
CUST001235 03/22/2019 01/05/2024   $8,945.00 I 5559876543 jane.doe@email.com
CUST001236 06/10/2021 .            $3,250.50 A 555123.... customer@email
CUST001237 11/30/2018 11/15/2023  $45,892.00 A 5555551212 vip.customer@company.com
;
run;

/* Process and clean the data */
data processed_customers;
    set raw_customer_input;
    
    /* Set proper lengths and labels */
    length customer_category $20 
           email_domain $30
           phone_formatted $14;
           
    label cust_id = 'Customer ID'
          join_date = 'Membership Start Date'
          last_purchase = 'Most Recent Purchase'
          total_purchases = 'Lifetime Purchase Amount'
          status = 'Account Status'
          customer_category = 'Customer Tier';
    
    /* Calculate customer metrics */
    days_member = today() - join_date;
    years_member = intck('year', join_date, today(), 'C');
    
    /* Handle missing last purchase */
    if missing(last_purchase) then do;
        days_since_purchase = .;
        purchase_status = 'Never Purchased';
    end;
    else do;
        days_since_purchase = today() - last_purchase;
        if days_since_purchase <= 90 then purchase_status = 'Active';
        else if days_since_purchase <= 180 then purchase_status = 'At Risk';
        else purchase_status = 'Inactive';
    end;
    
    /* Categorize customers */
    if total_purchases >= 25000 then customer_category = 'Platinum';
    else if total_purchases >= 10000 then customer_category = 'Gold';
    else if total_purchases >= 5000 then customer_category = 'Silver';
    else customer_category = 'Bronze';
    
    /* Format phone number */
    if length(compress(phone, '.')) = 10 then
        phone_formatted = cat('(', substr(phone,1,3), ') ', 
                             substr(phone,4,3), '-', 
                             substr(phone,7,4));
    else phone_formatted = 'Invalid';
    
    /* Extract email domain */
    email_domain = scan(email, 2, '@');
    
    /* Apply formats */
    format join_date last_purchase mmddyy10.
           total_purchases dollar12.2
           status $status_fmt.;
run;

/* Create analysis report */
proc freq data=processed_customers;
    tables customer_category purchase_status status;
    title "Customer Analysis Summary";
run;
```

### Example 2: Financial Data with Custom Formats

```sas
/* Create comprehensive financial formats */
proc format;
    /* Risk scoring format */
    value risk_score
        0 - 300 = 'Very High Risk'
        300 <- 500 = 'High Risk'
        500 <- 650 = 'Medium Risk'
        650 <- 750 = 'Low Risk'
        750 <- 850 = 'Very Low Risk'
        850 <- 999 = 'Minimal Risk'
        . = 'Not Scored';
    
    /* Account type format */
    value $acct_type
        'CHK' = 'Checking'
        'SAV' = 'Savings'
        'MMA' = 'Money Market'
        'CD' = 'Certificate of Deposit'
        'IRA' = 'Individual Retirement'
        'LOC' = 'Line of Credit'
        other = 'Unknown Type';
    
    /* Transaction format */
    value $trans_cat
        'DEP' = 'Deposit'
        'WTH' = 'Withdrawal'
        'FEE' = 'Service Fee'
        'INT' = 'Interest'
        'TRF' = 'Transfer'
        'PMT' = 'Payment';
    
    /* Picture format for account numbers */
    picture acctnum
        0-999999999 = '000-00-0000' (prefix='ACCT: ');
run;

/* Apply formats to financial data */
data financial_analysis;
    input account_num account_type $ balance 
          credit_score trans_type $ amount;
    
    /* Calculate risk metrics */
    if balance < 0 then balance_risk = 'Overdrawn';
    else if balance < 100 then balance_risk = 'Low Balance';
    else if balance < 1000 then balance_risk = 'Moderate';
    else balance_risk = 'Healthy';
    
    /* Apply all formats */
    format account_num acctnum.
           account_type $acct_type.
           balance dollar12.2
           credit_score risk_score.
           trans_type $trans_cat.
           amount dollar10.2;
    
    datalines;
123456789 CHK -50.00 . WTH 100.00
234567890 SAV 15000.00 725 DEP 500.00
345678901 MMA 50000.00 810 INT 125.50
456789012 LOC -2500.00 650 PMT 250.00
567890123 IRA 125000.00 790 TRF 5000.00
;
run;

proc print data=financial_analysis;
    title "Financial Account Analysis";
run;
```

## Practical Exercises

### Exercise 1: Data Type Precision

Create a program that demonstrates numeric precision issues and their solutions:

```sas
/* Your task: Complete this program */
data precision_test;
    /* Test 1: Create variables with different lengths */
    length small 3 medium 5 large 8;
    
    /* Assign the same large number to each */
    small = _______; 
    medium = _______;
    large = _______;
    
    /* Test 2: Demonstrate floating point issues */
    calc1 = 0.1 + 0.2;
    calc2 = 0.3;
    
    /* Create a safe comparison */
    are_equal = _______;
    
    /* Test 3: Find maximum safe integer for each length */
    /* Hint: Use loops and test when precision is lost */
run;
```

**Solution:**
```sas
data precision_test;
    /* Test 1: Create variables with different lengths */
    length small 3 medium 5 large 8;
    
    /* Assign the same large number to each */
    test_number = 123456789;
    small = test_number; 
    medium = test_number;
    large = test_number;
    
    /* Test 2: Demonstrate floating point issues */
    calc1 = 0.1 + 0.2;
    calc2 = 0.3;
    exact_equal = (calc1 = calc2);  /* May be FALSE! */
    
    /* Create a safe comparison */
    are_equal = (round(calc1, 0.0000001) = round(calc2, 0.0000001));
    
    /* Test 3: Find maximum safe integer for each length */
    array lengths[3] small3 medium5 large8 (3 5 8);
    array max_safe[3];
    
    do i = 1 to 3;
        if lengths[i] = 3 then max_safe[i] = 8192;
        else if lengths[i] = 5 then max_safe[i] = 8388608;
        else max_safe[i] = 9007199254740992;
    end;
    
    put "Small (3 bytes): " small= comma20.;
    put "Medium (5 bytes): " medium= comma20.;
    put "Large (8 bytes): " large= comma20.;
    put "0.1 + 0.2 = 0.3? " exact_equal=;
    put "Safe comparison: " are_equal=;
run;
```

### Exercise 2: Custom Format Creation

Create custom formats for a retail business:

```sas
/* Your task: Create formats for:
   1. Product categories (codes A-E)
   2. Price ranges for products
   3. Customer loyalty levels based on points
   4. Season codes (SP, SU, FA, WI)
*/

proc format;
    /* Add your format definitions here */
run;

/* Test your formats with this data */
data retail_test;
    input product_code $ price points season $;
    datalines;
A 29.99 1500 SP
C 149.99 3200 SU
E 9.99 750 FA
B 79.99 5000 WI
;
run;
```

**Solution:**
```sas
proc format;
    /* Product categories */
    value $prod_cat
        'A' = 'Accessories'
        'B' = 'Clothing'
        'C' = 'Electronics'
        'D' = 'Home Goods'
        'E' = 'Clearance'
        other = 'Unknown Category';
    
    /* Price ranges */
    value price_range
        low - 19.99 = 'Budget'
        20 - 49.99 = 'Standard'
        50 - 99.99 = 'Premium'
        100 - high = 'Luxury';
    
    /* Customer loyalty levels */
    value loyalty
        0 - 999 = 'Bronze Member'
        1000 - 2499 = 'Silver Member'
        2500 - 4999 = 'Gold Member'
        5000 - high = 'Platinum Member'
        . = 'Not a Member';
    
    /* Season codes */
    value $season_fmt
        'SP' = 'Spring Collection'
        'SU' = 'Summer Collection'
        'FA' = 'Fall Collection'
        'WI' = 'Winter Collection'
        other = 'Year-Round';
run;

/* Test your formats with this data */
data retail_test;
    input product_code $ price points season $;
    
    format product_code $prod_cat.
           price price_range.
           points loyalty.
           season $season_fmt.;
    
    datalines;
A 29.99 1500 SP
C 149.99 3200 SU
E 9.99 750 FA
B 79.99 5000 WI
;
run;

proc print data=retail_test;
    title "Retail Data with Custom Formats";
run;
```

### Exercise 3: Date Calculations

Create a program that calculates various date-related metrics:

```sas
/* Your task: Complete these calculations */
data date_metrics;
    /* Input employee data */
    input emp_id hire_date :mmddyy10. birth_date :mmddyy10.;
    
    /* Calculate:
       1. Exact age in years
       2. Years of service
       3. Retirement date (65 years old)
       4. Days until retirement
       5. Next work anniversary
       6. Quarter when hired
    */
    
    datalines;
101 03/15/2020 06/10/1985
102 07/01/2018 12/25/1990
103 01/30/2022 09/05/1975
;
run;
```

**Solution:**
```sas
data date_metrics;
    /* Input employee data */
    input emp_id hire_date :mmddyy10. birth_date :mmddyy10.;
    
    /* Calculate exact age in years */
    age_exact = intck('year', birth_date, today(), 'C');
    
    /* Years of service */
    years_service = intck('year', hire_date, today(), 'C');
    
    /* Retirement date (65 years old) */
    retirement_date = intnx('year', birth_date, 65, 'same');
    
    /* Days until retirement */
    days_to_retire = retirement_date - today();
    
    /* Next work anniversary */
    /* Calculate years since hire, add 1, then find that anniversary */
    years_employed = intck('year', hire_date, today());
    next_anniversary = intnx('year', hire_date, years_employed + 1, 'same');
    
    /* Handle case where anniversary already passed this year */
    if next_anniversary < today() then
        next_anniversary = intnx('year', hire_date, years_employed + 2, 'same');
    
    /* Quarter when hired */
    hire_quarter = qtr(hire_date);
    hire_qtr_year = cat(year(hire_date), '-Q', hire_quarter);
    
    /* Format dates */
    format hire_date birth_date retirement_date next_anniversary mmddyy10.;
    
    datalines;
101 03/15/2020 06/10/1985
102 07/01/2018 12/25/1990
103 01/30/2022 09/05/1975
;
run;

proc print data=date_metrics;
    title "Employee Date Metrics";
run;
```

## Common Pitfalls and Solutions

### 1. Character Variable Truncation

```sas
/* Problem: Default length truncates data */
data truncation_problem;
    /* First occurrence sets length */
    if type = 1 then category = 'A';  /* Length becomes 1 */
    else category = 'Category B';      /* Truncated to 'C' */
run;

/* Solution: Always declare length first */
data truncation_solution;
    length category $20;
    if type = 1 then category = 'A';
    else category = 'Category B';      /* Full value stored */
run;
```

### 2. Format vs. Informat Confusion

```sas
/* Problem: Using format when informat needed */
data wrong_usage;
    /* This won't work as expected */
    input date mmddyy10.;  /* This is correct - informat */
    date = mmddyy10.;      /* This is wrong - format in assignment */
run;

/* Solution: Use informats for input, formats for output */
data correct_usage;
    input date :mmddyy10.;        /* Informat for reading */
    format date mmddyy10.;        /* Format for display */
    
    /* Or use INPUT function */
    date_string = '01/15/2024';
    date_value = input(date_string, mmddyy10.);  /* INPUT function with informat */
run;
```

### 3. Missing Value Propagation

```sas
/* Problem: Missing values in calculations */
data missing_problem;
    x = 10;
    y = .;
    z = 5;
    
    /* These all become missing */
    sum1 = x + y + z;
    avg1 = (x + y + z) / 3;
run;

/* Solution: Use SAS functions */
data missing_solution;
    x = 10;
    y = .;
    z = 5;
    
    /* These handle missing properly */
    sum2 = sum(x, y, z);      /* Returns 15 */
    avg2 = mean(x, y, z);     /* Returns 7.5 */
    n_valid = n(x, y, z);     /* Returns 2 */
run;
```

## Best Practices Summary

### 1. Variable Declaration

```sas
/* Best practice: Declare all attributes upfront */
data well_structured;
    /* Length declarations */
    length customer_id $10 
           name $50 
           amount 8;
    
    /* Attributes with everything */
    attrib order_date length=8 format=mmddyy10. label='Order Date'
           status length=$1 format=$status_fmt. label='Order Status';
    
    /* Now start processing */
    set raw_data;
    /* ... */
run;
```

### 2. Format Organization

```sas
/* Best practice: Centralize format definitions */
libname library '/shared/formats';

proc format library=library.formats;
    /* Define all formats in one place */
run;

options fmtsearch=(library.formats work);
```

### 3. Date Handling Standards

```sas
/* Best practice: Consistent date handling */
%let date_format = mmddyy10.;
%let datetime_format = datetime20.;

data standardized_dates;
    set raw_data;
    
    /* Always use same format */
    format all_dates &date_format
           all_datetimes &datetime_format;
    
    /* Consistent date creation */
    today = today();
    month_start = intnx('month', today, 0, 'B');
    month_end = intnx('month', today, 0, 'E');
run;
```

## Summary and Key Takeaways

You've now mastered the essential details of SAS data handling:

### 1. **Data Types**
- Only two types: numeric and character
- Numeric precision depends on storage length
- Character variables need explicit length declaration
- Always consider precision in calculations

### 2. **Informats vs. Formats**
- Informats read data INTO SAS
- Formats control how SAS DISPLAYS data
- Custom formats are powerful for categorization
- PROC FORMAT creates reusable formats

### 3. **Variable Attributes**
- LENGTH controls storage and precision
- LABEL provides documentation
- FORMAT controls display
- Attributes can be inherited

### 4. **Missing Values**
- Regular (.) and special (.A-.Z) missing values
- SAS functions handle missing values gracefully
- Always check for missing values in critical calculations
- Document missing value meanings

### 5. **Dates and Times**
- SAS dates are numeric (days since 1/1/1960)
- Rich set of date functions and formats
- INTCK counts intervals, INTNX increments dates
- Consistent date handling prevents errors

## What's Next?

In Part 5 of our series, "Data Input and Output," we'll explore:
- Reading various file formats (CSV, Excel, fixed-width)
- Advanced INFILE options
- Writing data to external files
- PROC IMPORT and EXPORT
- Handling special characters and delimiters

## Additional Resources

1. **SAS Documentation**: [Formats and Informats](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/leforinforref/titlepage.htm)

Remember: Attention to data types, formats, and proper handling of missing values and dates will save you hours of debugging and ensure your analyses are accurate and professional.

Happy formatting! You're building the skills that separate good SAS programmers from great ones.