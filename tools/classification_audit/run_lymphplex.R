# Call the unmodified author algorithm with numeric missing values preserved.
args <- commandArgs(TRUE)
stopifnot(length(args) == 3)
suppressPackageStartupMessages({library(cluster); library(dplyr); library(reshape2)})
source(file.path(args[1], 'Functions.R'))
centers <- read.delim(file.path(args[1], 'Centers.txt'), check.names=FALSE)
seeds <- read.delim(file.path(args[1], 'Seeds.txt'), check.names=FALSE)
features <- names(centers)[-1]
x <- read.delim(args[2], check.names=FALSE, na.strings=c('', 'NA'), colClasses='character')
stopifnot(all(c('Sample', features, 'TP53') %in% names(x)), !anyDuplicated(x$Sample))
# All-missing CSV columns otherwise become logical, which reshape2::melt treats as IDs.
x[,c(features,'TP53')] <- lapply(x[,c(features,'TP53')], as.numeric)
stopifnot(!anyNA(x$TP53), all(x$TP53 %in% c(0,1)))
stopifnot(all(as.matrix(x[,features]) %in% c(0,1,NA)))
labels <- rep('TP53', nrow(x))  # Author TP53 override has priority.
remaining <- which(x$TP53 == 0)
if (length(remaining)) {
    stopifnot(all(rowSums(!is.na(x[remaining,features,drop=FALSE])) > 0))
    assigned <- suppressMessages(GetCluster(x[remaining,c('Sample',features),drop=FALSE], centers, seeds))
    stopifnot(identical(as.character(assigned[,1]), x$Sample[remaining]))
    labels[remaining] <- as.character(assigned[,2])
}
stopifnot(all(labels %in% c(as.character(centers[,1]), 'TP53')))
write.table(data.frame(Sample=x$Sample,call=labels),args[3],sep='\t',quote=FALSE,row.names=FALSE)
