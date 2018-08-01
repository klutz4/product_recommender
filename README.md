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

Parameters used for all:
* Subset_size = 35,000
* No tokenizer used for any vectorizer

Parameters used for Cosine Similarity:
* TfIdfVectorizer

Parameters used for LDA:
* CountVectorizer
* batch_size = 100
* max_iter = 10

Parameters used for MiniBatchKMeans:
* TfIdfVectorizer
* n_clusters = 50
* batch_size = 100 (default)


### The Results

Comparing the three methods with the dataset restricted to the 'art' category:

``` python
Please enter the index of your item (up to 236706): 9490
Your chosen item is 'Blury Style' Graphic Art Print on Wrapped Canvas, which costs $101.99


What is your price range?
 (Please enter your range as min-max): 50-200
Would you like to use Cosine Sim, LDA, or Kmeans? Kmeans
How many recommendations would you like? 3
Mini Batch KMeans:

Recommending 3 products similar to 'Blury Style' Graphic Art Print on Wrapped Canvas...
-------
Recommended: 'Foggy Days (254)' Photographic Print on Canvas
Price: $136.99
Recommended: 'Winter Feeling (27)' Photographic Print on Canvas
Price: $82.99
Recommended: 'Stone(21)' Photographic Print on Canvas
Price: $157.99
```
Our chosen item:  
![alt text](https://secure.img1-fg.wfcdn.com/im/75284972/resize-h400-w400%5Ecompr-r85/5248/52488516/%27Blury+Style%27+Graphic+Art+Print+on+Wrapped+Canvas.jpg)

Cosine Sim Recommendations:  
<img src = 'https://secure.img1-fg.wfcdn.com/im/42344136/resize-h400-w400%5Ecompr-r85/2340/23405252/%27Waxwings+by+Dmitry+Dubikovskiy+Graphic+Art+Print.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/95491154/resize-h400-w400%5Ecompr-r85/5358/53581353/%27California+Living%27+Photographic+Print+on+Canvas.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/79227590/resize-h400-w400%5Ecompr-r85/5189/51895085/%27Bords+Gris%27+Framed+Watercolor+Painting+Print.jpg' width=275>

LDA Recommendations:  
<img src = 'https://secure.img1-fg.wfcdn.com/im/89100010/resize-h400-w400%5Ecompr-r85/2927/29270827/%22W.D.+Clark+Plane+C%22+by+Cole+Borders+Graphic+Art+on+Wrapped+Canvas.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/23434750/resize-h400-w400%5Ecompr-r85/4728/47282237/Diligence+Graphic+Art+on+Wrapped+Canvas.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/91596251/resize-h400-w400%5Ecompr-r85/3183/31838960/%27Gem%27+Graphic+Art+on+Plaque.jpg' width=275>

KMeans Recommendations:  
<img src = 'https://secure.img1-fg.wfcdn.com/im/50416156/resize-h400-w400%5Ecompr-r85/5778/57786729/%27Foggy+Days+%28254%29%27+Photographic+Print+on+Canvas.jpg' width=275>
<img src ='https://secure.img1-fg.wfcdn.com/im/17872227/resize-h400-w400%5Ecompr-r85/5917/59172587/%27Winter+Feeling+%2827%29%27+Photographic+Print+on+Canvas.jpg' width=275>
<img src ='https://secure.img1-fg.wfcdn.com/im/36562174/resize-h400-w400%5Ecompr-r85/5867/58671817/%27Stone%2821%29%27+Photographic+Print+on+Canvas.jpg' width=275>


Another one!
Chosen item:
<img src ='https://secure.img1-fg.wfcdn.com/im/22355347/resize-h400-w400%5Ecompr-r85/3980/39804222/%27Richmond+Virginia+Skyline%27+Graphic+Art+Print+on+Canvas.jpg'>

Cosine Sim Recommendations:  
<img src = 'https://secure.img1-fg.wfcdn.com/im/42344136/resize-h400-w400%5Ecompr-r85/2340/23405252/%27Waxwings+by+Dmitry+Dubikovskiy+Graphic+Art+Print.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/95491154/resize-h400-w400%5Ecompr-r85/5358/53581353/%27California+Living%27+Photographic+Print+on+Canvas.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/79227590/resize-h400-w400%5Ecompr-r85/5189/51895085/%27Bords+Gris%27+Framed+Watercolor+Painting+Print.jpg' width=275>

LDA Recommendations:  
<img src = 'https://secure.img1-fg.wfcdn.com/im/89100010/resize-h400-w400%5Ecompr-r85/2927/29270827/%22W.D.+Clark+Plane+C%22+by+Cole+Borders+Graphic+Art+on+Wrapped+Canvas.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/23434750/resize-h400-w400%5Ecompr-r85/4728/47282237/Diligence+Graphic+Art+on+Wrapped+Canvas.jpg' width=275>
<img src = 'https://secure.img1-fg.wfcdn.com/im/91596251/resize-h400-w400%5Ecompr-r85/3183/31838960/%27Gem%27+Graphic+Art+on+Plaque.jpg' width=275>

KMeans Recommendations:  
<img src = 'https://secure.img1-fg.wfcdn.com/im/48090541/resize-h400-w400%5Ecompr-r85/3980/39801890/%27Travel+Poster+17%27+Graphic+Art+Print+on+Canvas.jpg' width=275>
<img src ='https://secure.img1-fg.wfcdn.com/im/07227074/resize-h400-w400%5Ecompr-r85/4063/40635151/%27Foggy+Morning+at+Sea%27+Graphic+Art+Print+on+Wrapped+Canvas.jpg' width=275>
<img src ='https://secure.img1-fg.wfcdn.com/im/06698891/resize-h490-w490%5Ecompr-r85/5748/57482275/%27The+Balloon+Man%27+Photographic+Print+on+Wrapped+Canvas.jpg' width=275>


### Future Work

* Try to cluster and label the 'unmapped - misclassified' products.
* Obtain a dataset with more of the unrepresented categories.
* Use neural networks to incorporate image processing to improve the labels and recommendations.


### References

Special thanks to Bill Sherby and the people at Havenly for allowing me to work with their data.
