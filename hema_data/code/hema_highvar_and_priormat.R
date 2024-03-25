library(data.table)
library(stringr)

clean_pathreg_data <- function(file_name){
  file_loc <- paste0("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/raw_data/",
                     file_name, ".csv")
  B_12 <- fread(file_loc, sep = "\t")
  B_12 <- B_12[-(1:10)] # Remove the first 10 rows
  B_12 <- B_12[1:(.N - 20)]# Remove the last 20 rows
  B_12 <- B_12[order(dpt_pseudotime)]
 
  all_genes <- setdiff(names(B_12), "dpt_pseudotime")
  #B_12[,time_lab := paste0("t_", 1:.N)]
  
  B_12 <- melt(B_12, measure.vars = all_genes,
               id.vars = c("dpt_pseudotime"),
               value.name = "expression",
               variable.name = "gene")
  B_12 <- B_12[!is.na(gene) & gene != "", ]
  setnames(B_12, 
           old = c("dpt_pseudotime", "expression"),
           new = c("time_point", "exprs"))
  B_12[, sample:= file_name]
  
  
  return(B_12)
}

min_max_normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}


### will want genes to also have prior info so...
long_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_clean_harmonized_full_prior.csv")
all_prior_affiliated_genes <- unique(c(long_prior$from, long_prior$to))

### start cleaning


data_names <- c("eryth", "mono", "B")

for (lineage in data_names){
  
  A_12 <- clean_pathreg_data(paste0(lineage,"_21_norm"))
  B_12 <- clean_pathreg_data(paste0(lineage,"_24_norm"))
  C_12 <- clean_pathreg_data(paste0(lineage,"_36_norm"))
  
  full_data <- rbind(A_12, B_12, C_12)
  print(full_data[, .(my_var = var(exprs)), by = .(gene)][order(-my_var)][1:10,])
  full_data[, c("time_point", "exprs") := {
    list(
      min_max_normalize(time_point),
      min_max_normalize(exprs)
    )
  }, by = sample]
  full_data[is.na(exprs), exprs := 0]
  
  
  all_genes <- full_data[, unique(gene)]
  all(all_prior_affiliated_genes %in% all_genes)
  
  num_genes <-  length(all_genes) 
  print(num_genes)
  full_data <- full_data[gene %in% all_genes]
  
  full_data[, length(unique(gene)), by = .(sample)]
  
  
  time_fake_gene <- unique(full_data[,.(sample, time_point)])
  time_fake_gene[, gene:= "ZZZ_time_col"]
  time_fake_gene[, exprs:= time_point]
  
  full_data_melt <- merge(full_data, time_fake_gene, 
                          by = c("sample","time_point","gene","exprs"),
                          all = T)
  
  full_data_melt[, avail_time := paste("avail",sprintf("%04d", 1:.N), sep = "_"), 
                 by = .(sample, gene)]
  
  
  datamat <- dcast(full_data_melt,
                   sample + gene ~ avail_time,
                   value.var = "exprs")
  
  datamat[, gene:= NULL]
  
  num_samples <- datamat[, length(unique(sample))]
  datamat[, sample:= NULL]
  top_row <- as.list(rep(NA, dim(datamat)[2]))
  top_row[[1]] <- num_genes
  top_row[[2]] <- num_samples
  datamat<- rbind(top_row, datamat)
  
  save_name = paste0( "C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/clean_data/hema_",
                      lineage,
                      "_529genes_",
                      num_samples,
                      "testsamples.csv")
  write.table(datamat,
              save_name, 
              sep=",",
              row.names = FALSE,
              col.names = FALSE,
              na = "")
}




write.csv(data.frame(gene = all_genes) , 
          "C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/clean_data/hema_gene_names_529.csv",
          row.names = F)



## Now make relevant prior
edges <- long_prior
rm(long_prior)
genes <- data.table(gene = all_genes)
setnames(genes, "gene", "x")
genes[, gene_sl := .I]

edges <- merge(edges, genes, by.x = "from", by.y = "x")
edges[, from := gene_sl]
edges[, gene_sl:= NULL]

edges <- merge(edges, genes, by.x = "to", by.y = "x")
edges[, to := gene_sl]
edges[, gene_sl:= NULL]

edges[,activation := 1, 
      by = .(to, from)]

edge_mat <- matrix(0, nrow = nrow(genes), ncol = nrow(genes))

update_edge_mat <- function(edge_mat, from, to, activation){
  edge_mat[from, to] <<- activation #effect of row on column
}
edges[,
      update_edge_mat(edge_mat,from, to, activation), 
      by= .(from, to)]

write.table(edge_mat,
            "C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/clean_data/edge_prior_matrix_hema_529.csv", 
            sep = ",",
             row.names = F, 
          col.names = F)


