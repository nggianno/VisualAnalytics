## VisualAnalytics - 2AMV10

###Disappearance at Gastech Challenge

## How to run this app

We suggest you to create a virtual environment for running this app with Python 3. Clone this repository 
and open your terminal/command prompt in the root folder.

open the command prompt
cd into the folder where you want to save the files and run the following commands. To get the HTTPS link, press the clone button in the right top corner and then copy the link under "Clone with HTTPS". 

```
> git clone <HTTPS link>
> cd <folder name on your computer>
> python -m venv venv

```
If python is not recognized use python3 instead

In Windows: 

```
> venv\Scripts\activate

```
In Unix system:
```
> source venv/bin/activate
```

Install all required packages by running:
```
> pip == 21.2.4
> pip install -r requirements.txt
```
Note: We noticed a small issue with requirements.txt and the automatic download of the packages. If there is a problem when replicating the code, just ```pip install``` packages manually.
Run this app locally with:
```
> python dashboard.py
```
You will get a http link, open this in your browser to see the results. You can edit the code in any editor (e.g. PyCharm) and if you save it you will see the results in the browser.

## Resources

* [Dash](https://dash.plotly.com/)
* [Plotly](https://plotly.com/python/)
