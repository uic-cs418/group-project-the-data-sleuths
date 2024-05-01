## Problem: Is Bird Diversity in Chicago Increasing?

The project explores the bird diversity across neighbourhoods in Chicago.It analyses eBird data, finds patterns and creates a machine learning model to predict the future bird diversity values of communities in Chicago.

### How do you measure bird diveersity?

Bird diversity is measured using a concept called Shannon Index. The Shannon Index, also known as Shannon Diversity Index or Shannon-Wiener Index, is a measure of biodiversity that quantifies the uncertainty in predicting the identity of a randomly selected species from a community. It takes into account both the species richness (number of different species) and the evenness (distribution of individuals among species) within the community.  
Higher values of the Shannon Index indicate greater species diversity within the community, with a maximum value occurring when all species are equally abundant. It is widely used in ecology and conservation biology to assess and compare biodiversity in different ecosystems

### How to run the project?

- Clone the repo and and checkout to the folder.
- The final notebook file is **eBirdCleanorgFinal.ipynb**. Open it in the browser using jupyter or any IDE like VS Code.
- The [initial dataset](https://drive.google.com/file/d/1Kcr8Ib8Vqu_Bg0E0ess7Zyeb-5FqJkiI/view?usp=drive_link) was too large(unzipped version : **>13 GB**). Hence the dataset was split into several small files(xaa,xab,xac) for easy processing using [split](https://man7.org/linux/man-pages/man1/split.1.html) command. The split files can be found at [link](https://drive.google.com/file/d/1cx8rGEnnR-s3eBecdr62qfHgJwJkWr8B/view?usp=sharing). Download the zip file, extract it to the data folder in the project directory before running Section 1.Keep in mind that the Section 1 execution takes approximately 1 hour on normal laptops.If you prefer to bypass the final working dataset creation process(**Section 1**), download the dataset directly using [link](https://drive.google.com/file/d/1hmlXERz6M9B25s_ogpSJU86cXNG-jW2v/view?usp=drive_linkhttps://drive.google.com/file/d/1hmlXERz6M9B25s_ogpSJU86cXNG-jW2v/view?usp=drive_link),save it in **data/** folder and run from the **Section 2**
