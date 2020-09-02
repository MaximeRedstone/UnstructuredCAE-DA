# Unstructured Convolutional Autoencoders for Big Data Assimilation - Imperial College London MSc Research Project

Data Assimilation (DA) provides a framework to incorporate observations into forecasting models thereby improving predictions. Many attempts have been made to improve the efficiency and accuracy of this process. Techniques such as Principal Component Analysis (PCA) or Truncated Singular Value Decomposition (TSVD) have been used to reduce the space. However, this de facto entails a loss of information. 

This repository implements the use of deep learning techniques and particularly Convolutional Autoencoders (CAEs) on unstructured meshes following a localisation procedure to reduce the space without any loss of information in the context of a Big Data Problem dealing with sub-domains in the 3D unstructured mesh.

It extends the implementation of CAEs for structured meshes found at: https://github.com/julianmack/Data_Assimilation

# Getting Started

**Installing Procedure**
```
git clone https://github.com/MaximeRedstone/UnstructuredCAE-DA.git
cd UnstructuredCAE-DA
pip install -r requirements.txt
```

**Repository Structure**

-*Localised1DCAE.py*: to reduce the space using CAEs on the result of the Localisation by Clustering in DA process\
-*LocalisedTSVD.py*: to reduce the space using TSVD on the result of the Localisation by Clustering in DA process

-**UnstructuredMesh**\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Unstructured CLIC Settings\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Localisation Loader Settings\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Data loader for CAEs\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Localiser to conduct localisation by clustering\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Adapted Tucodec Architectures, Residual Blocks
 
 -**Experiments**: results saved in either of the following directories:\
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-TSVD / **Experiment Title** / Domain1 / Training and Testing Loss and Time (csv)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DA results (csv)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;vtu / contains sample of grids where assimilation took place\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Domain 2 /\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...
							   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-CAE / **Experiment Title** / Domain1 /\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Domain 2 /\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...
							  
-**X_6** contains pickled data for sub-domain 6 and similarly for other sub-domains

-**LocalisationResults** contains visualisation of localisation clustering procedure

-**data**, **train**, **nn**, **AEs**, **settings**, **VarDA**, **utils**, **ML_utils** are taken from https://github.com/julianmack/Data_Assimilation with modifications to fit our extensions.

# Running

Examples of runs can be found in files *loc1DCAE.txt* and *tsvd.txt*.

```
python3 Localised1DCAE.py\
--title Experiment_Title\
--all_data_dir /path/to/data/dir/of/first/subdomain/   /path/to/data/dir/of/next/subdomain/  /path/to/data/dir/of/last/subdomain\
--pickled_dir /path/to/where/pickled/clustered/data/is/saved/\
--percentage Percentage_Of_Locations_Required\
--model_name 1D2L\
--retrain Boolean_Retrain_AE\
--mean_hist_data Boolean_DA_Background_State_Is_Mean_Of_Historical_Data
```

Additional Requirement: VTU data files are set to match following format: LSBU_TIMESTEP_SUBDOMAIN.vtu\
Overloading function get_sorted_fps_U in DataLoaderUnstructuredMesh could be required.
