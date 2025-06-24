# thesis
Spatiotemporal infectious Disease modelling
This repository is for my final year project work. I am working on estimating and inferring parameters from a spatio-temporal disease model. In this work, we investigated spatio-temporal dynamics of the 2014–2016 Ebola Virus Disease outbreak in West Africa (Guinea, Liberia, Sierra Leone) using SEIR compartmental models implemented in Stan.  We implemented a non-spatial SEIR model to capture temporal dynamics within Patches, after which we implemented a metapopulation model incorporating spatial coupling to assess inter-country transmissions. The models were fitted using Bayesian framework in stan to estimate parameters such as the transmission rate ($\beta$), progression rate ($\sigma$), recovery rate ($\gamma$) and the basic reproduction number ($R_0$). 

$R_0$ was used to compare the two models to assess the impact of spatial heterogeneity in disease modelling. Results show that spatial heterogeneity affects transmission rate in a given patch, hence the need to account for it in modelling infectious diseases.

Supervisors: Dr. Juliette Unwin and Dr. Michael Whitehouse 
