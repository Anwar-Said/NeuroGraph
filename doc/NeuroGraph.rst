NeuroGraph is a collection of graph-based neuroimaging datasets that span multiple categories of demographics, mental states and cognitive traits. The following provides an overview of these categories and their associated datasets.

The data is made available in accordance with the WU-Minn HCP Consortium Open Access Data Use Terms (Step 4), which can be found at  https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms.

Demographics
--------------------------

Demographics category includes gender and age estimation. The gender attribute facilitates a binary classification with the categories being male and female. Age is categorized into three distinct groups as in: 22-25, 26-30, and 31-35 years.  We introduce four datasets named: HCP-Gender, HCP-Age, DynHCP-Gender, and DynHCP-Age under this category. The first two are
static graph datasets while the last two are the corresponding dynamic graph datasets.

Mental States
-------------------------------
The mental state decoding involves seven tasks: Emotion Processing, Gambling, Language, Motor, Relational Processing, Social Cognition, and Working Memory. Each task is designed to help delineate a core set of functions relevant to different facets of the relation between human brain, cognition and behavior. Under this category, we present two datasets: HCP-Activity, a static representation, and DynHCP-Activity, its dynamic counterpart.

Cognitive Traits
-----------------------------------
The cognitive traits category of our dataset comprises two significant traits: working memory (List Sorting) and fluid intelligence evaluation with PMAT24. Working memory refers to an individual’s capacity to temporarily hold and manipulate information, a crucial aspect that influences higher cognitive functions such as reasoning, comprehension, and learning. Fluid intelligence represents the ability to solve novel problems, independent of any knowledge from the past. It demonstrates the capacity to analyze complex relationships, identify patterns, and derive solutions in dynamic situations. The prediction of both these traits, quantified as continuous variables in our dataset, are treated as regression problem. We aim to predict
the performance or scores related to these cognitive traits based on the functional connectome graphs. We generate four datasets under cognitive traits: HCP Fluid Intelligence (HCP-FI), HCP Working Memory (HCP-WM), DynHCP-FI and DynHCP-WM.
