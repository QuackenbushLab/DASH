library(data.table)
library(stringr)


### will want genes to also have prior info so...
long_ppi <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_clean_harmonized_PPI.csv")
long_prior <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/otter_clean_harmonized_full_prior.csv")

all_prior_affiliated_genes <- unique(c(long_prior$from, long_prior$to))
all_ppi_affiliated_genes <- unique(c(long_ppi$from, long_ppi$to))
mean(!all_ppi_affiliated_genes %in% all_prior_affiliated_genes)
### so 3.6% of ppi affiliated genes are not part of the prior, let's remove these 

long_ppi[, genes_exist_in_motif:= from %in% all_prior_affiliated_genes &
           to %in% all_prior_affiliated_genes, by = .(from, to)]
long_ppi <- long_ppi[genes_exist_in_motif == T, ]

all_ppi_affiliated_genes <- unique(c(long_ppi$from, long_ppi$to))
mean(!all_ppi_affiliated_genes %in% all_prior_affiliated_genes)
#Good

gene_names <- fread("C:/STUDIES/RESEARCH/ODE_project_local_old/hema_data/clean_data/gene_names_hema.csv",
                    header = T)
names(gene_names) <- "gene"
num_genes <- 529 #basically all that the prior has
high_var_genes <- gene_names$gene


## Now make relevant PPi
edges <- long_ppi
genes <- high_var_genes
setnames(gene_names, "gene", "x")
gene_names[, gene_sl := .I]

edges <- merge(edges, gene_names, by.x = "from", by.y = "x")
edges[, from := gene_sl]
edges[, gene_sl:= NULL]

edges <- merge(edges, gene_names, by.x = "to", by.y = "x")
edges[, to := gene_sl]
edges[, gene_sl:= NULL]

edges[,activation := value, 
      by = .(to, from)]

edge_mat <- matrix(0, nrow = nrow(gene_names), ncol = nrow(gene_names))

update_edge_mat <- function(edge_mat, from, to, activation){
  edge_mat[from, to] <<- activation #effect of row on column
}
edges[,
      update_edge_mat(edge_mat,from, to, activation), 
      by= .(from, to)]

write.table(edge_mat,
            "C:/STUDIES/RESEARCH//ODE_project_local_old/hema_data/clean_data/PPI_matrix_hema_529.csv",
            sep = ",",
            row.names = F, 
            col.names = F
)

