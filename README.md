# Citation 

MLA:

	Nilforoshan, Hamed, and Neil Shah. "SliceNDice: Mining Suspicious Multi-attribute Entity Groups with Multi-view Graphs." IEEE DSAA (2019).

bibtex:

    @article{nilforoshan2019slicendice, 
    title={SliceNDice: Mining Suspicious Multi-attribute Entity Groups with Multi-view Graphs},
    author={Nilforoshan, Hamed and Shah, Neil},
    journal={IEEE DSAA},
    year={2019}
    }





# Background
This repository contains code for SliceNDice, an approach to discover clusters of similar entities which behave 
synchronously across possibly multiple features/views, using the multi-view graph mining approach proposed in 
[*SliceNDice: Mining Suspicious Multi-attribute Entity Groups with Multi-view Graphs  (Nilforoshan & Shah, 2019)*](https://drive.google.com/open?id=1aByTr2fPm_Z8AVDpyvE8paAVdpR6B4Ns).  The approach accepts 
as input a dataframe of shape *(n_entities, n_views + 1)*, using 1 column as an entity identifier, where the cells of 
the dataframe are sets of strings representing possibly multiple attribute values associated with the entity on that 
view.  The approach also takes as input a parameter *Z* specifying the number of views on which to find synchronous 
entity groups; basically, it guides the algorithm to try to optimize synchronicity of sets of entities across 
![](https://latex.codecogs.com/gif.latex?n_%7Bviews%7D%20%5Cchoose%20Z) view subsets.  The algorithm works by first 
building a set of network representations across entities, where each view yields its own network, and the edges between
 entities in each network view are weighted according to (a) how many attribute values they share, and (b) how rare 
 the attribute values are.  The intuition is that a group of entities that is behaving suspiciously similarly will be a 
 "dense" subnetwork in this representation (i.e. have highly concentrated edge weight, due to many similarities which 
 are rare in the underlying dataset).  The algorithm will automatically discover such groups, and rank them according 
 to the statistical (un)likelihood of observing a given subnetwork with some size and mass across the chosen views, 
 using the background subnetworks as effective "null" models.

There are numerous applications for such an approach in community detection, fraud detection and network summarization.
We show one application in which we deploy SliceNDice in the ad integrity setting, where we discover groups of 
collusive advertiser organizations which may be scheming to engage in user-related or platform-related fraud; below, we 
show an example of our approach, and a network visualization of discovered fraudsters in the Snapchat ads platform:

![](./ad_integrity/example.png)

# Deploying
``./deploy.sh`` takes care of installing dependencies and running the [driver](./driver.py).
The driver reads and processes JSON data into a Pandas dataframe, runs the approach to discover
subnetworks and writes metadata (network visualizations and shared asset files).

The input JSON data is assumed to have one line per entity, and one string field per attribute, such that
the set of (values per attribute per entity) is stored as a single CSV string.  For example, to denote an entity
"1" with attributes "country" and "IP" with 1+ values per attribute, we would use a single line:
`{"entity_id": "1", "country": "Libya,Germany,USA", "IP":"1.2.3.4"}`.
