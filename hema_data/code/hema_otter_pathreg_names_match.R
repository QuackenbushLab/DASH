library(data.table)
library(stringr)

### clean otter tf names to align with hemaedt
otter_motif_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/cancer_breast_otter_motif.csv")
otter_tf_names <- data.table(tf = setdiff(names(otter_motif_prior),"Row"))

hema_gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/clean_data/gene_names_hema.csv",
                            header = T)


counter <- 0
otter_gene_matcher <- function(this_name){
  counter <<- counter+1
  if (counter%%100 ==0){message(counter)}
  
  my_regex <- paste0("^",this_name,"$|",
                     "^",this_name," ///|",
                     "^/// ",this_name)
  hema_names <- unique(grep(my_regex, hema_gene_names$gene, value = T)) 
  if(this_name %in% hema_names){
    hema_names <- c(this_name)
  }
  n_hema_names <- length(hema_names)
  names_str <- paste0(hema_names,  collapse = " , ")
  return(list(n_hema_names, names_str))
}

otter_tf_names[ ,c("n_hema_names","hema_names"):= otter_gene_matcher(tf),
                by = tf]

write.csv(otter_tf_names, "C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_tf_names_hema_conv.csv",
          row.names = F)

## clean otter gene names to match with hemaedt
otter_gene_names <- data.table(read.delim("C:/STUDIES/RESEARCH/ODE_project_local_old/breast_cancer_data/ensemble_id_to_symbol.txt",
                                          header = F))
setnames(otter_gene_names,
         old = c("V1","V2"),
         new = c("ensemble_id", "gene"))

counter <- 0
otter_gene_names[ ,c("n_hema_names","hema_names"):= otter_gene_matcher(as.character(gene)),
                by = ensemble_id]

otter_gene_names[n_hema_names > 1, 
  hema_names := trimws((strsplit(hema_names, " , ")[[1]])[1]), 
  by = ensemble_id]

write.csv(otter_gene_names, "C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_gene_names_hema_conv.csv",
          row.names = F)
