# Havenly

### The Goal

* Build a recommender that will take one product and recommend similar products, in type, style and price

### The Data and Feature Engineering

For this project, I worked with a sample of the product data from Havenly, which consisted of ~300,000 rows with the following columns:

```python
['vendor_variant_id', 'vendor_id', 'product_title',
 'product_description', 'vendor_name', 'taxonomy_name', 'taxonomy_id',
 'weblink', 'color', 'material', 'pattern', 'is_returnable',
 'ship_surcharge', 'is_assembly_required', 'is_feed', 'commission_tier',
 'inventory_type', 'division', 'category', 'price', 'sale_price']
 ```
My first task was seeing how many null values were in the data and figuring out how I wanted to deal with them.

<img src = 'images/nullplot.png' width=1000>
The yellow represents the null values.

 Steps taken to clean the data:
 * Drop any columns comprised entirely of NaNs
 * Drop columns that wouldn't be used for clustering
 * Drop rows without a price or sale price
 * Fill the null values of the sale_price columns with the item's price (i.e. this item is not on sale)
 * Fill any null values with 'other' in categorical columns
 * Drop any remaining columns that still have null values ('brand_id','sku','upc','size','dimensions','image_url')

 <img src = 'images/category_prop.png'>

|Category|      Proportion|
|----|----|
|art  |.810013
|unmapped - mis-classified |  0.034453
|occassional seating  |0.022277
|sofas & sectionals  |0.021371
|lighting | 0.020269
|unmapped - low priority category | 0.020056
|bedroom furniture | 0.019625
|decorative accessories | 0.013058
|rugs|  0.007279|
|dining furniture | 0.006498
|windows | 0.005171
|bedding | 0.004743
|pillows | 0.003313
|outdoor furniture | 0.003015
|accent furniture | 0.001971
|bath | 0.001605
|office furniture | 0.001530
|mirrors | 0.001485
|media furniture | 0.000808
|storage furniture | 0.000797
|wallpaper | 0.000517
|other | 0.000147

 As shown above, my sample is primarily art, which makes recommending for products outside of that category difficult. I performed the clustering and recommending using all categories to start and then restricted to only those in the 'art' category.

### The Clustering

Since my data contains product titles, product descriptions and certain product features, I combined all columns with string values into one column, named 'combo', in order to use NLP for clustering.

I chose 4 initial clustering methods to try:

* NLP (TfIdfVectorizer) + cosine similarity
* Latent Dirichlet Allocation + cosine similarity
* MiniBatchKMeans
* Hierarchical clustering

The hierarchical clustering made for some cool looking plots...
<img src = 'images/dendrogram.png' width=1000>
But didn't prove to be much help.

I decided to limit my clustering comparison to the other three methods.

Once I had my methods, I had to find a way to incorporate the price restraints with my clusters. After all, you wouldn't want to plan on spending $100 on a chair and have a $1000 chair recommended to you. For this, I added a user input to specify the desired price range.

Fun fact: the most expensive product in my sample is a crystal chandelier for a whopping $19,045.00.

![alt text](https://static.havenly.com/product/production/php_5953ec1775e65.jpg)

Tuning the parameters:
max_iter
tokenizer
max_features (Kmeans)
batch size (LDA and Kmeans)

### The Results

Comparing the three methods:

``` python
Please enter the index of your item (up to 236706): 90490
Your chosen item is 'Theodore Roosevelt, Secretary of Navy' by Forbes Litho. Mfg. Co. Memorabilia, which costs $559.99


What is your price range?
 (Please enter your range as min-max): 500-800
Would you like to use Cosine Sim, LDA, or Kmeans? Cosine Sim
How many recommendations would you like? 5
NLP and Cosine Similarity:

Recommending 5 products similar to 'Theodore Roosevelt, Secretary of Navy' by Forbes Litho. Mfg. Co. Memorabilia...
-------
Recommended: 'Will You Supply Eyes for the Navy?' by Gordon Grant Vintage Advertisement
Price: $536.99
(Cosine similarity: 0.2508)
Recommended: 'Navy Shorts' Framed Graphic Art Print
Price: $529.99
(Cosine similarity: 0.1818)
Recommended: 'Bikini' Framed Graphic Art Print in Navy Blue
Price: $789.99
(Cosine similarity: 0.1699)
Recommended: 'Thurston: the Great Magician' by Strobridge Litho. Co Vintage Advertisement
Price: $559.99
(Cosine similarity: 0.1363)
Recommended: 'Newmann's Wonderful Spirit Mysteries' by Donaldson Litho. Co Vintage Advertisement
Price: $569.99
(Cosine similarity: 0.1325)
```
Our chosen item:  
![alt text](https://secure.img1-fg.wfcdn.com/im/75284972/resize-h400-w400%5Ecompr-r85/5248/52488516/%27Blury+Style%27+Graphic+Art+Print+on+Wrapped+Canvas.jpg)

Cosine Sim Recommendations:  
![alt text]()
![alt text]()
![alt text]()

LDA Recommendations:  
![alt text]()
![alt text]()
![alt text]()

KMeans Recommendations:  
![alt text](https://secure.img1-fg.wfcdn.com/im/13197713/resize-h400-w400%5Ecompr-r85/5259/52594267/%27Blury+Style%27+Graphic+Art+Print+on+Wrapped+Canvas.jpg)
![alt text](https://secure.img1-fg.wfcdn.com/im/67962037/resize-h400-w400%5Ecompr-r85/5166/51669942/%27Abstract+Style%27+Graphic+Art+Print+on+Wrapped+Canvas.jpg)
![alt text](https://secure.img1-fg.wfcdn.com/im/98825325/resize-h400-w400%5Ecompr-r85/5914/59147794/%27Street+Life+184%27+Photographic+Print+on+Canvas.jpg)


### Future Work

* Try to cluster and label the 'unmapped - misclassified' products.
* Obtain a dataset with more of the unrepresented categories.
* Use neural networks to incorporate image processing to improve the labels and recommendations.


### References

Special thanks to Bill Sherby and the people at Havenly for allowing me to work with their data.
