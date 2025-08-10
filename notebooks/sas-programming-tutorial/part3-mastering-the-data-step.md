# Part 3: Mastering the DATA Step - The Heart of SAS Programming

Welcome to Part 3 of our comprehensive SAS programming tutorial series! Now that you understand the fundamentals of SAS and the difference between DATA and PROC steps, it's time to dive deep into the DATA step—the true powerhouse of SAS programming. This is where data transformation magic happens!

## What You'll Learn

In this tutorial, you will:
- Understand the DATA step's compilation and execution phases
- Master the Program Data Vector (PDV) concept
- Learn various INPUT statement techniques for reading data
- Work with automatic variables (_N_ and _ERROR_)
- Control data flow with IF-THEN-ELSE logic
- Create multiple datasets from a single DATA step
- Apply real-world data manipulation techniques

## Prerequisites

- Completion of Part 1: Getting Started with SAS
- Completion of Part 2: SAS Fundamentals
- Basic understanding of DATA vs. PROC steps
- Access to SAS (SAS OnDemand, SAS Studio, or any SAS installation)

## The DATA Step Architecture: Behind the Scenes

Understanding how SAS processes a DATA step is crucial for writing efficient programs. Every DATA step goes through two distinct phases: compilation and execution.

### Compilation Phase vs. Execution Phase

Think of the DATA step like preparing and performing a musical piece:
- **Compilation Phase**: Like rehearsing—checking the sheet music (syntax), assigning instruments (variables), and planning the performance
- **Execution Phase**: The actual performance—playing each note (processing each observation) according to the plan

Let's visualize this process:

```sas
/* Example to demonstrate compilation vs execution */
data employee_info;
    /* During COMPILATION: SAS checks syntax and creates PDV */
    input emp_id name $ salary;
    
    /* During EXECUTION: These statements run for each observation */
    annual_salary = salary * 12;
    bonus = annual_salary * 0.10;
    
    datalines;
101 John 5000
102 Mary 6000
103 Bob 5500
;
run;

/* Let's trace what happens */
proc print data=employee_info;
    title "Results after DATA Step Processing";
run;
```

### What Happens During Compilation?

1. **Syntax Checking**: SAS verifies all statements are valid
2. **Variable Attributes**: Determines variable types, lengths, and formats
3. **PDV Creation**: Sets up the Program Data Vector
4. **Execution Plan**: Creates a roadmap for processing

### What Happens During Execution?

1. **Initialize PDV**: Set all variables to missing
2. **Read Data**: Input one observation
3. **Execute Statements**: Process each statement sequentially
4. **Output**: Write to output dataset (if conditions are met)
5. **Repeat**: Return to step 1 for next observation

## The Program Data Vector (PDV): SAS's Working Memory

The PDV is like a workbench where SAS assembles each observation before writing it to the output dataset. Understanding the PDV is key to mastering DATA step programming.

### Visualizing the PDV

```sas
/* Let's trace through the PDV */
data trace_pdv;
    /* Show PDV contents at various stages */
    input x y;
    
    /* Statement 1: Before calculation */
    put "Before calc: " _all_;
    
    z = x + y;
    
    /* Statement 2: After calculation */
    put "After calc: " _all_;
    
    datalines;
10 20
30 40
;
run;
```

**Log Output Analysis:**
```
Before calc: x=10 y=20 z=. _ERROR_=0 _N_=1
After calc: x=10 y=20 z=30 _ERROR_=0 _N_=1
Before calc: x=30 y=40 z=. _ERROR_=0 _N_=2
After calc: x=30 y=40 z=70 _ERROR_=0 _N_=2
```

### PDV Components

The PDV contains:
1. **User-defined variables**: Variables you create
2. **Automatic variables**: _N_ (observation counter) and _ERROR_ (error flag)
3. **Temporary values**: Used during processing but not output

## Automatic Variables: _N_ and _ERROR_

SAS provides two automatic variables that are incredibly useful for debugging and control:

### _N_: The Observation Counter

```sas
/* Using _N_ to track processing */
data track_observations;
    input product $ price quantity;
    
    /* Add observation number to output */
    obs_number = _N_;
    
    /* Process only first 5 observations */
    if _N_ <= 5 then do;
        total = price * quantity;
        put "Processing observation " _N_ ": " product= total=;
    end;
    
    /* Stop after 10 observations */
    if _N_ > 10 then stop;
    
    datalines;
Apple 1.50 100
Banana 0.75 200
Orange 2.00 150
Grape 3.50 80
Peach 2.50 90
Mango 4.00 60
Pear 1.75 110
Plum 2.25 70
Kiwi 1.25 120
Berry 5.00 40
Melon 6.00 30
Lemon 0.50 180
;
run;
```

### _ERROR_: The Error Detection Flag

```sas
/* Demonstrating _ERROR_ variable */
data check_errors;
    input name $ age income;
    
    /* This will cause _ERROR_=1 for invalid data */
    if _ERROR_ then do;
        put "ERROR in observation " _N_ ": " name= age= income=;
        /* You might want to set values or flag bad records */
        data_quality_flag = 'Bad';
    end;
    else data_quality_flag = 'Good';
    
    datalines;
John 25 50000
Jane thirty 60000
Bob 35 seventy5k
Mike 28 55000
;
run;

proc print data=check_errors;
    title "Data Quality Check Results";
run;
```

## Creating Datasets: Multiple INPUT Techniques

The INPUT statement is your gateway to bringing external data into SAS. Let's explore the various methods:

### 1. List Input (Free Format)

The simplest form—values separated by spaces:

```sas
/* List input - simple but limited */
data list_input;
    input name $ age height weight;
    datalines;
John 25 5.9 170
Mary 30 5.4 125
Robert 28 6.1 185
;
run;
```

**Limitations:**
- Character values can't contain spaces
- Values must be separated by at least one space
- Missing values must be represented by periods

### 2. Column Input (Fixed Columns)

Reads data from specific column positions:

```sas
/* Column input - great for fixed-width data */
data column_input;
    input name $ 1-10 
          age 11-13 
          salary 14-20
          dept $ 21-25;
    datalines;
John Smith 25 50000 SALES
Mary Jones 30 60000 IT   
Bob Miller 28 55000 HR   
;
run;

proc print data=column_input;
    title "Column Input Results";
run;
```

**Advantages:**
- Handles embedded spaces
- No delimiters needed
- Can read data in any order

### 3. Formatted Input

Reads data using informats:

```sas
/* Formatted input - for special data types */
data formatted_input;
    input @1 name $10. 
          @11 hire_date mmddyy10. 
          @22 salary dollar10.
          @33 phone $12.;
    format hire_date mmddyy10. salary dollar10.2;
    datalines;
John Smith01/15/2020 $55,000  555-123-4567
Mary Jones06/20/2019 $60,000  555-987-6543
Bob Miller03/10/2018 $58,500  555-456-7890
;
run;

proc print data=formatted_input;
    title "Formatted Input with Special Data Types";
run;
```

### 4. Mixed Input Styles

Combine different input methods:

```sas
/* Mixed input - best of all worlds */
data mixed_input;
    input name $15. @20 age 2. @25 (q1-q4) (3.);
    array quarters{4} q1-q4;
    
    /* Calculate annual total */
    annual_total = sum(of quarters{*});
    
    datalines;
John Smith      25   100200150175
Mary Johnson    30   125225175200
Robert Brown    28   150175225250
;
run;

proc print data=mixed_input;
    title "Mixed Input Style Results";
run;
```

### 5. Reading Multiple Records per Observation

Sometimes one observation spans multiple lines:

```sas
/* Multiple records per observation */
data employee_details;
    input #1 emp_id name $15.
          #2 address $30.
          #3 phone $12. email $25.;
    datalines;
101 John Smith
123 Main Street, Anytown USA
555-123-4567 john.smith@email.com
102 Mary Johnson
456 Oak Avenue, Somewhere City
555-987-6543 mary.j@email.com
103 Robert Brown
789 Pine Road, Elsewhere Town
555-456-7890 rbrown@email.com
;
run;

proc print data=employee_details;
    title "Multi-Line Input Results";
run;
```

## Variable Management in DATA Steps

### Variable Types and Attributes

```sas
/* Comprehensive variable management */
data variable_demo;
    /* Set lengths explicitly */
    length first_name $20 
           last_name $20 
           full_name $41
           employee_id 8
           status $1;
    
    /* Apply formats and labels */
    format hire_date mmddyy10.
           salary dollar12.2
           commission percent8.2;
           
    label employee_id = "Employee ID Number"
          full_name = "Employee Full Name"
          hire_date = "Date of Hire"
          salary = "Annual Salary"
          commission = "Commission Rate"
          status = "Employment Status";
    
    /* Input data */
    input employee_id first_name $ last_name $ 
          hire_date :mmddyy10. salary commission status $;
    
    /* Create derived variables */
    full_name = catx(' ', first_name, last_name);
    years_employed = intck('year', hire_date, today());
    
    datalines;
1001 John Smith 01/15/2018 55000 0.05 A
1002 Mary Johnson 06/20/2019 60000 0.07 A
1003 Robert Brown 03/10/2017 58000 0.06 I
;
run;

/* View the structure */
proc contents data=variable_demo;
    title "Variable Attributes in Dataset";
run;
```

### Working with Missing Values

```sas
/* Handling missing values effectively */
data missing_values;
    input name $ test1 test2 test3;
    
    /* Calculate average, handling missing values */
    
    /* Method 1: Using MEAN function (ignores missing) */
    avg_score = mean(test1, test2, test3);
    
    /* Method 2: Custom calculation with missing check */
    if nmiss(test1, test2, test3) = 0 then
        custom_avg = (test1 + test2 + test3) / 3;
    else if nmiss(test1, test2, test3) < 3 then do;
        sum_scores = sum(test1, test2, test3);
        n_scores = n(test1, test2, test3);
        custom_avg = sum_scores / n_scores;
    end;
    else custom_avg = .;
    
    /* Flag records with missing values */
    if nmiss(test1, test2, test3) > 0 then 
        missing_flag = 'Y';
    else 
        missing_flag = 'N';
    
    datalines;
John 85 90 88
Mary 78 . 82
Bob . . 75
Sue 92 88 .
Tom 88 85 90
;
run;

proc print data=missing_values;
    title "Missing Value Handling Examples";
run;
```

## Control Flow: IF-THEN-ELSE Logic

### Basic Conditional Processing

```sas
/* Comprehensive IF-THEN-ELSE example */
data sales_categorization;
    input salesperson $ region $ sales;
    
    /* Multiple conditions with ELSE IF */
    length performance $10 bonus_rate 8;
    
    if sales >= 100000 then do;
        performance = 'Excellent';
        bonus_rate = 0.15;
    end;
    else if sales >= 75000 then do;
        performance = 'Good';
        bonus_rate = 0.10;
    end;
    else if sales >= 50000 then do;
        performance = 'Average';
        bonus_rate = 0.05;
    end;
    else do;
        performance = 'Below';
        bonus_rate = 0.02;
    end;
    
    /* Calculate bonus */
    bonus = sales * bonus_rate;
    
    /* Regional adjustment */
    if region = 'West' then bonus = bonus * 1.1;
    else if region = 'East' then bonus = bonus * 1.05;
    
    datalines;
John North 85000
Mary West 120000
Bob South 45000
Sue East 95000
Tom West 55000
Lisa North 110000
;
run;

proc print data=sales_categorization;
    title "Sales Performance Categorization";
    format sales bonus dollar10.2 bonus_rate percent8.1;
run;
```

### Subsetting IF vs. WHERE

Understanding the difference is crucial:

```sas
/* Demonstrating Subsetting IF vs WHERE */

/* Using Subsetting IF */
data subset_if;
    input name $ age income;
    
    /* Calculate tax (executes for all records) */
    if income > 50000 then tax = income * 0.25;
    else tax = income * 0.15;
    
    /* Subsetting IF - only output qualifying records */
    if age >= 25;
    
    datalines;
John 23 45000
Mary 28 60000
Bob 22 35000
Sue 30 70000
Tom 26 55000
;
run;

/* Using WHERE */
data subset_where;
    set subset_if;
    where income > 50000;  /* WHERE is more efficient for existing datasets */
run;

/* Compare the results */
proc print data=subset_if;
    title "Results with Subsetting IF (age >= 25)";
run;

proc print data=subset_where;
    title "Results with WHERE (income > 50000)";
run;
```

### Complex Conditional Logic

```sas
/* Advanced conditional processing */
data employee_classification;
    input emp_id name $ dept $ years_service salary;
    
    /* Complex categorization logic */
    length category $20 eligibility $30;
    
    /* Nested conditions */
    if dept = 'IT' then do;
        if years_service >= 5 and salary >= 70000 then
            category = 'Senior IT';
        else if years_service >= 3 then
            category = 'Mid IT';
        else
            category = 'Junior IT';
    end;
    else if dept = 'Sales' then do;
        if salary >= 80000 then
            category = 'Top Sales';
        else if salary >= 60000 then
            category = 'Mid Sales';
        else
            category = 'Entry Sales';
    end;
    else
        category = 'Other';
    
    /* Multiple condition checking */
    if years_service >= 5 and salary >= 60000 and dept in ('IT', 'Sales') then
        eligibility = 'Stock Options';
    else if years_service >= 3 and salary >= 50000 then
        eligibility = 'Bonus Program';
    else if years_service >= 1 then
        eligibility = 'Standard Benefits';
    else
        eligibility = 'Probationary';
    
    datalines;
101 John IT 6 75000
102 Mary Sales 4 65000
103 Bob IT 2 55000
104 Sue Sales 7 85000
105 Tom HR 3 50000
106 Lisa IT 1 48000
;
run;

proc print data=employee_classification;
    title "Complex Employee Classification";
run;
```

## Creating Multiple Datasets in One DATA Step

One powerful feature of the DATA step is creating multiple output datasets simultaneously:

```sas
/* Creating multiple datasets based on conditions */
data high_sales (drop=reason)
     low_sales (drop=bonus)
     problem_records;
     
    input store_id $ month sales employees;
    
    /* Calculate metrics */
    if employees > 0 then sales_per_emp = sales / employees;
    else sales_per_emp = .;
    
    /* Route to appropriate dataset */
    if sales_per_emp = . then do;
        reason = 'Missing employee count';
        output problem_records;
    end;
    else if sales_per_emp >= 10000 then do;
        bonus = sales * 0.02;
        output high_sales;
    end;
    else do;
        reason = 'Low productivity';
        output low_sales;
    end;
    
    datalines;
S001 1 50000 5
S002 1 80000 6
S003 1 30000 4
S004 1 . 3
S005 1 60000 0
S006 1 120000 8
S007 1 25000 5
;
run;

/* Check all three datasets */
proc print data=high_sales;
    title "High Performing Stores";
run;

proc print data=low_sales;
    title "Low Performing Stores";
run;

proc print data=problem_records;
    title "Problem Records";
run;
```

## Advanced DATA Step Techniques

### Using RETAIN Statement

The RETAIN statement keeps values across observations:

```sas
/* Running totals with RETAIN */
data running_totals;
    input date :mmddyy10. daily_sales;
    format date mmddyy10. daily_sales month_sales year_sales dollar10.2;
    
    /* Retain values across observations */
    retain month_sales 0 year_sales 0;
    
    /* Extract month and year */
    current_month = month(date);
    current_year = year(date);
    
    /* Initialize retained variables at start of period */
    if _n_ = 1 then do;
        retain prev_month prev_year;
        prev_month = current_month;
        prev_year = current_year;
    end;
    
    /* Check for new month */
    if current_month ne prev_month then do;
        month_sales = 0;
        prev_month = current_month;
    end;
    
    /* Check for new year */
    if current_year ne prev_year then do;
        year_sales = 0;
        month_sales = 0;
        prev_year = current_year;
    end;
    
    /* Update running totals */
    month_sales + daily_sales;  /* Sum statement automatically retains */
    year_sales + daily_sales;
    
    datalines;
01/01/2024 5000
01/02/2024 6000
01/31/2024 7000
02/01/2024 5500
02/02/2024 6500
12/30/2024 8000
12/31/2024 9000
01/01/2025 5000
;
run;

proc print data=running_totals;
    title "Running Sales Totals";
run;
```

### First. and Last. Processing

Process data by groups:

```sas
/* First and Last processing */
proc sort data=sashelp.class out=class_sorted;
    by sex age;
run;

data age_groups;
    set class_sorted;
    by sex age;
    
    /* Initialize at start of each sex group */
    if first.sex then do;
        count_in_group = 0;
        total_weight = 0;
        total_height = 0;
    end;
    
    /* Accumulate within group */
    count_in_group + 1;
    total_weight + weight;
    total_height + height;
    
    /* Output summary at end of each age within sex */
    if last.age then do;
        avg_weight = total_weight / count_in_group;
        avg_height = total_height / count_in_group;
        output;
    end;
    
    /* Keep only summary variables */
    keep sex age count_in_group avg_weight avg_height;
run;

proc print data=age_groups;
    title "Summary by Sex and Age Groups";
    format avg_weight avg_height 8.1;
run;
```

## Practical Examples: Real-World Scenarios

### Example 1: Data Cleaning and Standardization

```sas
/* Real-world data cleaning scenario */
data messy_customer_data;
    input customer_id 
          name $20. 
          phone $15. 
          email $30. 
          join_date $10.;
    datalines;
1001  john smith      (555)123-4567  JOHN.S@EMAIL.COM    01/15/2020
1002  MARY JONES      555.987.6543   mary@email.com      2020-06-20
1003  Bob Brown       555 456 7890   bob.brown@email     03/10/19
1004  sue williams    5551234567     SUE@EMAIL.COM       15-JAN-20
;
run;

data clean_customer_data;
    set messy_customer_data;
    
    /* Standardize name - proper case */
    name = propcase(name);
    
    /* Standardize phone - remove all non-digits then format */
    phone_digits = compress(phone, '0123456789', 'k');
    if length(phone_digits) = 10 then
        phone_formatted = cats('(', substr(phone_digits,1,3), ') ',
                              substr(phone_digits,4,3), '-',
                              substr(phone_digits,7,4));
    else phone_formatted = 'Invalid';
    
    /* Standardize email - lowercase and validate */
    email = lowcase(email);
    if index(email, '@') > 0 and index(email, '.') > 0 then
        email_valid = 'Y';
    else email_valid = 'N';
    
    /* Parse and standardize date */
    /* Try different date formats */
    if index(join_date, '/') > 0 then
        join_date_sas = input(join_date, mmddyy10.);
    else if index(join_date, '-') > 0 then do;
        if anyalpha(substr(join_date,1,2)) then
            join_date_sas = input(join_date, date9.);
        else
            join_date_sas = input(join_date, yymmdd10.);
    end;
    
    format join_date_sas mmddyy10.;
    
    /* Quality flags */
    if phone_formatted = 'Invalid' or email_valid = 'N' or missing(join_date_sas) then
        needs_review = 'Y';
    else
        needs_review = 'N';
    
    drop phone_digits;
run;

proc print data=clean_customer_data;
    title "Cleaned and Standardized Customer Data";
run;
```

### Example 2: Transaction Processing with Business Rules

```sas
/* Complex transaction processing */
data transactions;
    input trans_id customer_id trans_date :mmddyy10. 
          trans_type $ amount location $;
    format trans_date mmddyy10. amount dollar10.2;
    datalines;
101 1001 01/15/2024 PURCHASE 150.00 STORE
102 1001 01/15/2024 RETURN -50.00 STORE  
103 1002 01/16/2024 PURCHASE 200.00 ONLINE
104 1003 01/16/2024 PURCHASE 75.00 STORE
105 1002 01/17/2024 PURCHASE 125.00 STORE
106 1001 01/18/2024 PURCHASE 300.00 ONLINE
107 1003 01/18/2024 RETURN -75.00 STORE
108 1004 01/19/2024 PURCHASE 500.00 ONLINE
;
run;

/* Process transactions with business rules */
proc sort data=transactions;
    by customer_id trans_date;
run;

data customer_summary fraud_alerts;
    set transactions;
    by customer_id;
    
    /* Initialize customer counters */
    if first.customer_id then do;
        total_purchases = 0;
        total_returns = 0;
        purchase_count = 0;
        return_count = 0;
        first_trans_date = trans_date;
        high_value_flag = 'N';
        same_day_return = 'N';
    end;
    
    /* Track previous transaction for same-day detection */
    retain prev_trans_date prev_amount;
    
    /* Process based on transaction type */
    if trans_type = 'PURCHASE' then do;
        total_purchases + amount;
        purchase_count + 1;
        
        /* Flag high-value transactions */
        if amount >= 300 then high_value_flag = 'Y';
        
        /* Store for same-day return check */
        prev_trans_date = trans_date;
        prev_amount = amount;
    end;
    else if trans_type = 'RETURN' then do;
        total_returns + abs(amount);
        return_count + 1;
        
        /* Check for same-day return */
        if trans_date = prev_trans_date and abs(amount) = prev_amount then do;
            same_day_return = 'Y';
            output fraud_alerts;
        end;
    end;
    
    /* Calculate summary at customer level */
    if last.customer_id then do;
        net_amount = total_purchases - total_returns;
        days_active = trans_date - first_trans_date;
        return_rate = total_returns / total_purchases;
        
        /* Business rule: Flag suspicious activity */
        if return_rate > 0.5 or same_day_return = 'Y' then
            risk_flag = 'HIGH';
        else if return_rate > 0.3 or high_value_flag = 'Y' then
            risk_flag = 'MEDIUM';
        else
            risk_flag = 'LOW';
        
        output customer_summary;
    end;
    
    keep customer_id net_amount days_active return_rate risk_flag
         trans_id trans_date amount; /* Keep trans details for fraud_alerts */
run;

proc print data=customer_summary;
    title "Customer Transaction Summary";
    format net_amount dollar10.2 return_rate percent8.1;
run;

proc print data=fraud_alerts;
    title "Potential Fraud Alerts - Same Day Returns";
run;
```

## Practical Exercises

### Exercise 1: PDV Understanding

Complete this program to demonstrate PDV behavior:

```sas
/* Your task: Add PUT statements to show PDV contents */
data pdv_exercise;
    /* Initialize a retained variable */
    retain counter 0;
    
    input x y;
    
    /* Add PUT statement here to show PDV before calculations */
    
    counter + 1;
    z = x * y + counter;
    
    /* Add PUT statement here to show PDV after calculations */
    
    if z > 50 then output;
    
    datalines;
5 8
10 6
3 4
15 5
;
run;
```

**Solution:**
```sas
data pdv_exercise;
    retain counter 0;
    
    input x y;
    
    /* Show PDV before calculations */
    put "Before: " x= y= counter= z= _N_=;
    
    counter + 1;
    z = x * y + counter;
    
    /* Show PDV after calculations */
    put "After: " x= y= counter= z= _N_= /;
    
    if z > 50 then output;
    
    datalines;
5 8
10 6
3 4
15 5
;
run;
```

### Exercise 2: Multiple Input Methods

Read this fixed-format data using appropriate INPUT techniques:

```
Positions:
1-5:   ID
7-20:  Name  
22-29: Hire Date (MM/DD/YY)
31-37: Salary (with $)
39-41: Department code
```

Data:
```
10001 John Smith     01/15/20 $55,000 IT 
10002 Mary Johnson   06/20/19 $60,000 SAL
10003 Robert Brown   03/10/18 $58,500 HR 
```

**Solution:**
```sas
data employee_records;
    input id 1-5
          name $ 7-20
          @22 hire_date mmddyy8.
          @31 salary dollar7.
          dept $ 39-41;
    
    format hire_date mmddyy10. salary dollar8.0;
    
    datalines;
10001 John Smith     01/15/20 $55,000 IT 
10002 Mary Johnson   06/20/19 $60,000 SAL
10003 Robert Brown   03/10/18 $58,500 HR 
;
run;

proc print data=employee_records;
    title "Employee Records from Fixed Format";
run;
```

### Exercise 3: Complex Data Processing

Create a program that:
1. Reads student test scores
2. Calculates weighted average (Test1: 30%, Test2: 30%, Final: 40%)
3. Assigns letter grades
4. Creates separate datasets for passing and failing students
5. Includes a summary dataset with class statistics

**Solution:**
```sas
data passing(drop=needs_help)
     failing(drop=dean_list)
     class_summary(keep=total_students pass_count fail_count 
                        class_average highest_score lowest_score);
     
    input student_id name $ test1 test2 final;
    
    /* Calculate weighted average */
    weighted_avg = (test1 * 0.30) + (test2 * 0.30) + (final * 0.40);
    
    /* Assign letter grade */
    if weighted_avg >= 90 then letter_grade = 'A';
    else if weighted_avg >= 80 then letter_grade = 'B';
    else if weighted_avg >= 70 then letter_grade = 'C';
    else if weighted_avg >= 60 then letter_grade = 'D';
    else letter_grade = 'F';
    
    /* Special recognitions */
    if weighted_avg >= 95 then dean_list = 'Y';
    else dean_list = 'N';
    
    if weighted_avg < 60 then needs_help = 'Y';
    else needs_help = 'N';
    
    /* Output to appropriate dataset */
    if letter_grade in ('A','B','C') then output passing;
    else output failing;
    
    /* Accumulate class statistics */
    retain total_students 0 pass_count 0 fail_count 0
           sum_scores 0 highest_score 0 lowest_score 100;
    
    total_students + 1;
    sum_scores + weighted_avg;
    
    if letter_grade in ('A','B','C') then pass_count + 1;
    else fail_count + 1;
    
    if weighted_avg > highest_score then highest_score = weighted_avg;
    if weighted_avg < lowest_score then lowest_score = weighted_avg;
    
    /* Output summary at end */
    if eof then do;
        class_average = sum_scores / total_students;
        output class_summary;
    end;
    
    datalines;
1001 John 85 88 92
1002 Mary 78 75 80
1003 Bob 92 95 98
1004 Sue 65 60 58
1005 Tom 88 85 90
1006 Lisa 55 58 62
1007 Mike 95 98 96
1008 Ann 70 72 75
;
run;

proc print data=passing;
    title "Passing Students";
run;

proc print data=failing;
    title "Students Needing Additional Support";
run;

proc print data=class_summary;
    title "Class Summary Statistics";
run;
```

## Common Pitfalls and How to Avoid Them

### 1. Misunderstanding PDV Reinitialization

```sas
/* Problem: Expecting variables to retain values */
data wrong_accumulation;
    input sales;
    total + sales;  /* This works - sum statement retains */
    running_total = running_total + sales;  /* This doesn't work! */
    datalines;
100
200
300
;
run;

/* Solution: Use RETAIN or sum statement */
data correct_accumulation;
    input sales;
    retain running_total 0;  /* Explicitly retain */
    total + sales;           /* Sum statement auto-retains */
    running_total = running_total + sales;
    datalines;
100
200
300
;
run;
```

### 2. Incorrect WHERE vs. IF Usage

```sas
/* Problem: Using WHERE with calculated variables */
data problem;
    set sashelp.class;
    bmi = (weight / (height**2)) * 703;
    where bmi > 20;  /* ERROR: BMI doesn't exist yet */
run;

/* Solution: Use subsetting IF for calculated variables */
data solution;
    set sashelp.class;
    bmi = (weight / (height**2)) * 703;
    if bmi > 20;  /* Works with calculated variables */
run;
```

### 3. Missing RUN Statement Effects

```sas
/* Problem: DATA step continues unexpectedly */
data test1;
    x = 1;
    /* Missing RUN here */
    
data test2;  /* This becomes part of test1! */
    y = 2;
run;

/* Solution: Always include RUN statements */
data test1;
    x = 1;
run;

data test2;
    y = 2;
run;
```

## Best Practices for DATA Step Programming

### 1. Document Your Logic

```sas
data well_documented;
    set raw_data;
    
    /************************************************
    * Business Rule: Customer Segmentation          *
    * - VIP: Total purchases > $10,000             *
    * - Gold: Total purchases $5,000-$10,000       *
    * - Silver: Total purchases < $5,000           *
    ************************************************/
    
    if total_purchases > 10000 then segment = 'VIP';
    else if total_purchases >= 5000 then segment = 'Gold';
    else segment = 'Silver';
run;
```

### 2. Use Meaningful Variable Names

```sas
/* Good naming */
data customer_analysis;
    first_purchase_date = '01JAN2020'd;
    days_since_first_purchase = today() - first_purchase_date;
    is_active_customer = (days_since_last_purchase <= 90);
run;

/* Avoid cryptic names */
data bad_names;
    fpd = '01JAN2020'd;
    dsfp = today() - fpd;
    iac = (dslp <= 90);
run;
```

### 3. Initialize Variables Properly

```sas
data proper_initialization;
    /* Initialize all accumulator variables */
    retain total_amount 0 
           transaction_count 0
           error_count 0;
    
    /* Clear temporary variables */
    length temp_calc 8 error_msg $200;
    call missing(temp_calc, error_msg);
    
    /* Process data... */
run;
```

## Summary and Key Takeaways

Congratulations! You've now mastered the DATA step, the heart of SAS programming. Here's what you've learned:

### 1. **DATA Step Architecture**
- Compilation phase checks syntax and creates the PDV
- Execution phase processes observations one by one
- Understanding both phases helps write efficient code

### 2. **Program Data Vector (PDV)**
- The PDV is SAS's working memory for each observation
- Variables are reinitialized unless retained
- Automatic variables _N_ and _ERROR_ help with control

### 3. **INPUT Techniques**
- List input for simple space-delimited data
- Column input for fixed-position data
- Formatted input for special data types
- Mixed input combines the best of all methods

### 4. **Control Flow**
- IF-THEN-ELSE provides conditional processing
- Subsetting IF controls which observations to output
- WHERE is more efficient for filtering existing datasets

### 5. **Advanced Features**
- RETAIN keeps values across observations
- Multiple datasets can be created in one DATA step
- First. and Last. enable BY-group processing

### 6. **Best Practices**
- Always initialize retained variables
- Use meaningful variable names
- Document complex business logic
- Test with small datasets first

## What's Next?

In Part 4 of our series, "Variables, Formats, and Data Types," we'll explore:
- Deep dive into SAS data types and precision
- Mastering informats and formats
- Creating custom formats with PROC FORMAT
- Advanced date and time handling
- Working with missing values effectively

## Additional Resources

1. **Practice Dataset**: Download [data_step_exercises.zip](/assets/data/sas-tutorial/data_step_exercises.zip) for hands-on practice
2. **Quick Reference Card**: [DATA Step Quick Reference](/assets/data/sas-tutorial/data-step-reference.pdf)
3. **SAS Documentation**: [DATA Step Concepts](https://documentation.sas.com/doc/en/pgmsascdc/9.4_3.5/basess/titlepage.htm)

Remember: The DATA step is where the real power of SAS lies. The more you practice with different scenarios, the more comfortable you'll become with its capabilities. Don't be afraid to experiment—that's how you'll discover new techniques and solutions!

Happy DATA step programming! You're well on your way to becoming a SAS expert.