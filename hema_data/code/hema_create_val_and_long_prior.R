library(data.table)
library(stringr)

### Read in name files
otter_gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_gene_names_hema_conv.csv")
otter_tf_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_tf_names_hema_conv.csv")


#### Clean prior (OTTER)
otter_motif_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/cancer_breast_otter_motif.csv")
otter_motif_prior <- merge(otter_motif_prior, otter_gene_names,
                        by.x = "Row", by.y = "ensemble_id")
otter_motif_prior <- otter_motif_prior[n_hema_names >0 & hema_names != "",]

otter_motif_prior <- melt(otter_motif_prior,
                          id.vars = c("hema_names"),
                          measure.vars = otter_tf_names$tf,
                          )
otter_motif_prior <- otter_motif_prior[value == 1,]
otter_motif_prior <- otter_motif_prior[, .(tf = variable, to = hema_names)]
otter_motif_prior <- merge(otter_motif_prior, otter_tf_names,
                           by = "tf")
otter_motif_prior <- otter_motif_prior[n_hema_names >0 & hema_names != "",]
otter_motif_prior[,c("n_hema_names","tf"):=NULL]
otter_motif_prior <- otter_motif_prior[,.(from = hema_names, to = to)]

otter_motif_prior <- unique(otter_motif_prior)

write.csv(otter_motif_prior, "C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_clean_harmonized_full_prior.csv",
          row.names = F)


#### Clean PPI (OTTER) 
## Haven't changed anything below this yet (IH, 11/20/2023)
otter_PPI <- data.table(read.table("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/PPI_matrix_breast.txt"))
names(otter_PPI) <- otter_tf_names$tf
otter_PPI <- cbind(otter_tf_names, otter_PPI)

otter_PPI <- otter_PPI[n_hema_names >0 & hema_names != "",]

otter_PPI <- melt(otter_PPI,
                          id.vars = c("hema_names"),
                          measure.vars = otter_tf_names$tf)

otter_PPI <- otter_PPI[value > 0,]
otter_PPI <- otter_PPI[, .(tf = variable, to = hema_names, value = value)]
otter_PPI <- merge(otter_PPI, otter_tf_names,
                           by = "tf")
otter_PPI <- otter_PPI[n_hema_names >0 & hema_names != "",]
otter_PPI[,c("n_hema_names","tf"):=NULL]
otter_PPI <- otter_PPI[,.(from = hema_names, to = to, value = value)]

otter_PPI <- unique(otter_PPI)

write.csv(otter_PPI, "C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_clean_harmonized_PPI.csv",
          row.names = F)

