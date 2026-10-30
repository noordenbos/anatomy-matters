# Reproduce the author's reference fit; never fit on the query cohort.
args <- commandArgs(TRUE)
suppressPackageStartupMessages(library(flexmix))
reference <- read.csv(args[1], check.names=FALSE)
muts_df <- reference[,-1]
RNGversion('3.5.3'); set.seed(17)
ex <- initFlexmix(as.matrix(muts_df) ~ 1, k=1:10,
                 model=FLXMCmvbinary(), control=list(minprior=0.10), nrep=1000)
model <- getModel(ex, which='AIC')
saveRDS(list(aic=model, bic=getModel(ex,which='BIC'), fit=ex), args[2])
centers <- sapply(model@components, function(x) x[[1]]@parameters$center)
colnames(centers) <- paste0('reference_C',seq_len(ncol(centers)))
write.csv(centers, paste0(args[2],'.centers.csv'))
write.csv(data.frame(component=colnames(centers), prior=model@prior),paste0(args[2],'.priors.csv'),row.names=FALSE)
write.csv(data.frame(PID=reference$PID,component=paste0('reference_C',model@cluster)),paste0(args[2],'.calls.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=paste0(args[2],'.session.txt'))
