args <- commandArgs(trailingOnly=TRUE)
vendor <- args[1]; input <- args[2]; output <- args[3]
dir.create(output,recursive=TRUE,showWarnings=FALSE)
options(stringsAsFactors=FALSE)
source(file.path(vendor,'src/Model10_8.1.22.R'))
readcore <- function(name) read.table(file.path(vendor,'corefiles',name),header=TRUE,sep='\t')
samp <- read.table(file.path(input,'sample_annotation.txt'),header=TRUE,sep='\t')
flat <- read.table(file.path(input,'mutation_flat.txt'),header=TRUE,sep='\t')
genes <- read.table(file.path(input,'mutation_gene_list.txt'),header=TRUE,sep='\t')[[1]]
full <- as.matrix(readcore('Fullmat.2020.txt')==1)
study <- readcore('Study.samp.txt'); lab <- study[,18]; lab[lab=='SDT'] <- 'ST2'
feat <- readcore('Featord.2020.txt');feat[,11] <- feat[,11]==1
Index.Flat <- readcore('Index.Flat.txt')
result <- Predict9(flat,samp,Index.Flat,lab,
    Full.Uber.2020=readcore('Full.Uber.2020.txt'),Fullmat.2020=full,Featord.2020=feat,
    CGH.class=1,Has.Trunc=TRUE,mut.genelist=genes,CGH.genelist=NULL,
    Test.set=c('BN2','EZB','MCD','N1','ST2'),originalpred=readcore('originalpredv2.txt'),
    Has.CGH=study[,7]==1,CalcWilcox=FALSE,Version=2)
write.table(result$Prediction,file.path(output,'Result.tsv'),quote=FALSE,sep='\t',row.names=FALSE,na='NA')
write.table(result$Compare,file.path(output,'Compare.tsv'),quote=FALSE,sep='\t',row.names=FALSE,na='NA')
writeLines(as.character(result$Warn.set),file.path(output,'warnings.txt'))
