# Competition Approach Log
------

## Rough steps of approaching solution
------
1. Business Understanding - clearly define business problem and business oriented objectives of solution.
2. Data exploration - Understand each column by recording observations and characteristics of each column.
3. Data cleansing - Can be done simultaneously with step one. Removing duplicates, incorrect data and interpolate blank fields when needed.
4. Data exploration, - Using pandas to transform data parameters and visualization like matplotlib.
5. Modeling - Features engineering, model evaluation using gridsearch.
6. Submission to server for testing.


## Step 1 - Business Understanding
### Clearly define business problem and business oriented objectives of solution.
------
**Problem Definition**
Build product title quality models that can automatically grade product titles based on the provided features, and output two scores for each product:

**Type of Scores**
1. Clarity: a probability value between 0 and 1.  For example: 0.34139.
2. Conciseness: a probability value between 0 and 1.  For example: 0.98747.

**Clarity**
1 means within 5 seconds you can understand the title and what the product is. It easy to read. You can quickly figure out what its key attributes are (color, size, model, etc.).
0 means you are not sure what the product is after 5 seconds. it is difficult to read. You will have to read the title more than once.
**Conciseness**
1 means it is short enough to contain all the necessary information.
0 means it is too long with many unnecessary words. Or it is too short and it is unsure what the product is.

**Evaluation Metrics**
Submissions will be evaluated using the Root-Mean-Square Error (RMSE)


## Step 2 - Data exploration
### Understand each column by recording observations and characteristics of each column.
------
* country : The country where the product is marketed, with three possible values: my for Malaysia, ph for Philippines, sg for Singapore

* sku_id : Unique product id, e.g., "NO037FAAA8CLZ2ANMY"
   product identifier, check for duplicates

* title : Product title, e.g., "RUDY Dress"
   used as feature for clarity and Conciseness

* category_lvl_1 : General category that the product belongs to, e.g., "Fashion"

* category_lvl_2 : Intermediate category that the product belongs to, e.g., "Women"

* category_lvl_3 : Specific category that the product belongs to, e.g., "Clothing"

* short_description : Short description of the product, which may contain html formatting, e.g., "<ul> <li>Short Sleeve</li> <li>3 Colours 8 Sizes</li> <li>Dress</li> </ul>

* price : Price in the local currency, e.g., "33.0".  When country is my, the price is in Malaysian Ringgit.  When country is sg, the price is in Singapore Dollar.  When country is ph, the price is in Philippine Peso.

* product_type : It could have three possible values: local means the product is delivered locally, international means the product is delivered from abroad, NA means not applicable.

## Step 3 - Data cleansing

* removing html tags from fields

## Step 4 - Data exploration
TBC

## Step 5 - Modeling
TBC

## Step 6 - Submission

* Submitted to test server for results.

* Changed model parameters to improve results.









