#plot dates and locations
library("rnaturalearth")
library("ggplot2")
library(maps)       
library(mapdata)

meta <- read.table("data/allbrit.meta", as.is=TRUE)

###Dates
pdf("plots/date_brit.pdf",  width=2.24, height=2.24)
par(list(mar=c(4.4, 4.4, 0.4, 0.1), cex=0.5))
hist(meta[,2], xlim=c(5000,0), breaks=166, xlab="Date in Years BP", ylim=c(0,100), main=NA, border=NA, col=c("cyan", rep("blue", 400)))
dev.off()
 
###Locations
pdf("plots/map_brit.pdf",  width=2.24, height=2.24)
par(list(mar=c(4.4, 4.4, 0.4, 0.1), cex=0.5))
map('worldHires',c('UK', 'Ireland',  'Isle of Man','Isle of Wight', "Wales:Angelsey"),xlim=c(-11,4), ylim=c(49,60.9), mar=par("mar"))
points(meta[,5], meta[,4], pch=16, cex=0.5, col="blue")
dev.off()

###PCA
pca <- read.table("data/allbrit_HO.evec", as.is=TRUE)
ind <- read.table("data/allbrit.ind", as.is=TRUE)
rownames(meta) <- meta[,1]

kg <- c("CEU.SG", "GBR.SG", "TSI.SG", "IBS.SG")
kg.cols <- c( "blue", "cyan", "darkblue", "cornflowerblue")
names(kg.cols) <- kg

bg.cols <- kg.cols
names(bg.cols) <- c("Central", "Britain", "Italy", "Iberia") 

pca.ho <- pca[!(pca[,1]%in%ind[,1]),]
pca.pr <- pca[pca[,1]%in%ind[,1] & !(pca[,12]%in%kg),]
pca.kg <- pca[pca[,1]%in%ind[,1] & (pca[,12]%in%kg),]


tiff("plots/pca_brit.tiff", units="mm", width=57, height=57, res=300, compression="lzw")
par(list(mar=c(4.4, 4.4, 0.4, 0.1), cex=0.5))
p1 <- pca.kg[,]
p2 <- pca.pr[,]
plot(-pca.ho[,2], pca.ho[,3], col="grey", pch=16, cex=0.5, xlab="PC1", ylab="PC2")
points(-p2[,2], p2[,3], col="blue", pch=16, cex=0.5)
points(-p1[,2], p1[,3], col="cyan", pch=16, cex=0.5)
legend("topright", c("Reference", "Ancient", "Present-day"), pch=c(16,16,16), col=c("grey", "blue", "cyan"), bty="n")
dev.off()
