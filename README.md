# Unstructured Convolutional Autoencoders for Big Data Assimilation - Imperial College London MSc Research Project

Data Assimilation (DA) provides a framework to incorporate observations into forecasting models thereby improving predictions. Many attempts have been made to improve the efficiency and accuracy of this process. Techniques such as Principal Component Analysis (PCA) or Truncated Singular Value Decomposition (TSVD) have been used to reduce the space. However, this de facto entails a loss of information. 

This repository implements the use of deep learning techniques and particularly Convolutional Autoencoders (CAEs) on unstructured meshes following a localisation procedure to reduce the space without any loss of information in the context of a Big Data Problem dealing with sub-domains in the 3D unstructured mesh.

It extends the implementation of CAEs for structured meshes found at: https://github.com/julianmack/Data_Assimilation

# Getting Started

Installing Procedure: 

git clone https://github.com/MaximeRedstone/UnstructuredCAE-DA.git
cd UnstructuredCAE-DA
pip install -r requirements.txt

# Repository Structure:

-Localised1DCAE.py: to reduce the space using CAEs on the result of the Localisation by Clustering in DA process
-LocalisedTSVD.py: to reduce the space using TSVD on the result of the Localisation by Clustering in DA process

-\textbf{UnstructuredMesh}
	-Unstructured CLIC Settings
	-Localisation Loader Settings
	-Data loader for CAEs
	-Localiser to conduct localisation by clustering
	-Adapted Tucodec Architectures, Residual Blocks
 
 -\textbf{Experiments}: results saved in either of the following directories:
 	-TSVD / \textit{Experiment Title} / Domain1 / Training/ Testing Loss and Time (csv)
										           DA results (csv)
											   vtu / contains sample of grids where assimilation took place				
							   Domain 2 /
							   ...
							   
	-CAE / \textit{Experiment Title} / Domain1 / 
							 Domain 2 /
							  ...
							  
-\textbf{X_*} contains pickled data for sub-domain *

-\textbf{LocalisationResults} contains visualisation of localisation clustering procedure

-\textbf{data}, \textbf{train}, \textbf{nn}, \textbf{AEs}, \textbf{settings}, \textbf{VarDA}, \textbf{utils}, \textbf{ML_utils} are taken from https://github.com/julianmack/Data_Assimilation with modifications to fit our extension.

# Running

Examples of runs can be found in files loc1DCAE.txt and tsvd.txt.

python3 Localised1DCAE.py 	
--title Experiment_Title 	
--all_data_dir /path/to/data/dir/of/first/subdomain/   /path/to/data/dir/of/next/subdomain/  /path/to/data/dir/of/last/subdomain 	
--pickled_dir /path/to/where/pickled/clustered/data/is/saved/ 
--percentage Percentage_Of_Locations_Required
--model_name 1D2L
--retrain Boolean_Retrain_AE 	
--mean_hist_data Boolean_DA_Background_State_Is_Mean_Of_Historical_Data	

Additional Requirement: VTU data files are set to match following format: LSBU_TIMESTEP_SUBDOMAIN.vtu
Overloading function get_sorted_fps_U in DataLoaderUnstructuredMesh could be required.
