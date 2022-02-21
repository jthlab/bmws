library(ggplot2)
library(MASS)
library(fitdistrplus)
library(plotrix)
library(xtable)
source("mh_plot_lib.R")
source("scan_lib.R")

min.freq <- 0.1

cat("load\n")

data<-read.table("data/s_scan_all_brit.txt.gz", as.is=T, header=F)
colnames(data) <- c("CHR", "POS", "ID", "REF", "ALT", "FREQ", "S.HAT", "SL1", "SL2")
cat(paste0(sum(data$FREQ<=min.freq | data$FREQ>=(1-min.freq), na.rm=T), "\n"))
data <- data[data$FREQ>min.freq & data$FREQ<(1-min.freq),]
data <- data[!is.na(data$CHR),]
#p <- ggplot(data, aes(x=POS, y=SL1))+geom_point()+geom_vline(xintercept=136608646, color="red")

#redefine SL1 and 2
## data$SL1 <- data$S.HAT^2
## data$SL2 <- data$SL2^2-data$SL1

random<-read.table("data/s_scan_all_brit_random.txt.gz", as.is=T, header=F)
colnames(random) <- c("CHR", "POS", "ID", "REF", "ALT", "FREQ", "S.HAT", "SL1", "SL2")
random <- random[random$FREQ>min.freq & random$FREQ<(1-min.freq),]
random <- random[!is.na(random$CHR),]
## random$SL1 <- random$S.HAT^2
## random$SL2 <- random$SL2^2-random$SL1

cat("missing\n")
lmiss <- read.table("data/allbrit_ancientOnly.lmiss", as.is=TRUE, header=TRUE)
remove <- lmiss[lmiss$F_MISS>0.90,"SNP"]
data<-data[!(data$ID %in% remove),]
random<-random[!(random$ID %in% remove),]
cat(paste0(length(remove), "\n"))

frq <- read.table("data/allbrit_ancientOnly.frq", as.is=TRUE, header=TRUE)
remove <- frq[frq$MAF==0,"SNP"]
data<-data[!(data$ID %in% remove),]
random<-random[!(random$ID %in% remove),]
cat(paste0(length(remove), "\n"))

cat("window\n")

data.bin <- window.scan(data, bin.size=20, bin.skip=10)
random.bin <- window.scan(random, bin.size=20, bin.skip=10)

cat("params\n")
params1 <- fitdist(data.bin$SL1, distr = "gamma", method = "mme",lower = c(0, 0),  start=list(shape=10, rate=1000))
params2 <- fitdist(random.bin$SL1, distr = "gamma", method = "mme", lower = c(0, 0),  start=list(shape=10, rate=1000))
data.bin$P1 <- pgamma(data.bin$SL1, shape=params1$estimate[1], rate=params1$estimate[2], lower.tail=FALSE)
random.bin$P1 <- pgamma(random.bin$SL1, shape=params1$estimate[1], rate=params1$estimate[2], lower.tail=FALSE)
random.bin$P2 <- pgamma(random.bin$SL1, shape=params2$estimate[1], rate=params2$estimate[2], lower.tail=FALSE)
data.bin$P2 <- pgamma(data.bin$SL1, shape=params2$estimate[1], rate=params2$estimate[2], lower.tail=FALSE)

p.value.cutoff <- 7
p.value.cutoff.hi <- 7

cat("combine\n")
combine <- list()
ctr=1
for(chr in 1:22){
    cc <- data.bin[data.bin$CHR==chr,]
    cc <- cc[-log10(cc$P1)>p.value.cutoff,]
    cc <- cc[order(cc$P1),]
    while(NROW(cc)>0){
        combine[[ctr]] <- cc[1,,drop=FALSE]
        ctr=ctr+1
        cc <- cc[abs(cc$POS-cc[1,"POS"])>0.5e6,]
    }
}
combine <- do.call(rbind, combine)
cat("genes\n")
genes <- read.table("data/refseq.allgene.txt", as.is=TRUE, header=TRUE)
genes <- genes[!grepl("^MIR", genes$name),]
genes <- genes[!grepl("^ORF", genes$name),]
genes <- genes[!grepl("^SNORA", genes$name),]
genes <- genes[!grepl("^LOC", genes$name),]

combine$neargene<-NA
for(i in 1:NROW(combine)){
    these <- genes[genes$chr==combine[i,"CHR"],]
    dd <- pmax(0, these$start-combine[i,"POS"], combine[i,"POS"]-these$end)

    if(sum(c(dd==0))>0){
        gg <- unique(these[which(dd==0),"name"])
        if(length(gg)>2){
            ng <- paste0(c( gg[1], gg[2], "...") , collapse=",")
        }else{
            ng <- paste0(gg , collapse=",")
        }
    }else{
        ng <- unique(these[which.min(pmin(dd)),"name"])
    }
    
    combine[i,"neargene"] <- ng
}

combine[combine$CHR==2 & combine$POS>135e6 & combine$POS<137e6,"neargene"]<-"LCT"
#combine[combine$CHR==6 & combine$POS>25e6 & combine$POS<35e6,"neargene"]<-"HLA"
combine[combine$neargene=="RXFP3","neargene"] <- "SLC45A2"
combine[combine$neargene=="ZSCAN31","neargene"] <- "HLA1"
combine[combine$neargene=="HCG22","neargene"] <- "HLA2"
combine[combine$neargene=="PBX2","neargene"] <- "HLA3"

combine <- combine[!duplicated(combine$neargene),]
chr.starts <- get.chr.starts(data.bin$CHR,data.bin$POS)
combine$xpos <- trunc(combine$POS/100) + chr.starts[combine$CHR]

db <- data.frame(CHR=data.bin$CHR, POS=data.bin$POS, PVAL=data.bin$P1)

cat("plotting\n")

cplt <- combine
cplt <- cplt[!cplt$neargene%in%c("LINC00955", "HLA1", "HLA3"),]
cplt[cplt$neargene=="HLA2","neargene"] <- "HLA"

tiff("plots/scan_mh.tiff", units="mm", width=172, height=57, res=300, compression="lzw")
par(list(mar=c(2.5, 4.4, 0.4, 0.1), cex=0.5))
MH.plot(db)
text(cplt$xpos, -log10(cplt$P1), cplt$neargene, adj=-0.25, srt=90)
abline(h=p.value.cutoff, col="red", lty=3)
## abline(h=p.value.cutoff.hi, col="red", lty=3)
dev.off()

tiff("plots/scan_qq.tiff", units="mm", width=57, height=57, res=300, compression="lzw")
par(list(mar=c(4.4, 4.4, 0.4, 0.1), cex=0.5))
qqPlotOfPValues(data.bin$P1, cex.pts=2, xlim=c(0,6), ylim=c(0,20), col="darkblue" )
qqPlotOfPValues(random.bin$P2, cex.pts=2, xlim=c(0,6), col="darkred", add=T )
legend("topleft", c("Observed", "Randomized"), col=c("darkblue", "darkred" ), pch=16, bty="n")
abline(h=p.value.cutoff, col="red", lty=3)
## abline(h=p.value.cutoff.hi, col="red", lty=3)
dev.off()

#SDS comparison.
sds<-read.table("data/SDS_UK10K_n3195_release_Sep_19_2016.tab.gz", as.is=TRUE, header=TRUE)
sdsxdata<-merge(data, sds, by=c("CHR", "POS", "ID"))
sdsxdata <- sdsxdata[order(sdsxdata$CHR, sdsxdata$POS),]
combine$sds <- NA
for(i in 1:NROW(combine)){
    tx <- sdsxdata[sdsxdata$CHR==combine[i,"CHR"] & abs(sdsxdata$POS-combine[i,"POS"])<500000,"SDS"]^2
    combine[i,"sds"] <- mean(tx)
}
reg <- paste0(sdsxdata$CHR, "_", floor(sdsxdata$POS/1e6))
sdsdist <- aggregate(sdsxdata$SDS^2, mean, by=list(reg=reg))

cc <- combine[combine$sds>0 & combine$sds<10,]
rndanc <- floor(4*cc$sds)/4
labs<-aggregate(cc$neargene, by=list(rndanc), paste0, collapse=",")

labs <- labs[labs$x!="LINC00955",]

tiff("plots/scan_sds.tiff", units="mm", width=57, height=57, res=300, compression="lzw")
par(list(mar=c(4.4, 4.4, 0.4, 0.1), cex=0.5))
plot(density(sdsdist$x), xlim=c(0,10), col="darkblue", lwd=3, ylim=c(0,1.5), bty="n", xlab=expression("Mean SDS"^2), main="")
## hist(combine$sds, add=TRUE, col= alpha("red", 0.5), breaks=100, freq=FALSE)
## text(labs[,1]+0.18, 0.32*(1+nchar(gsub("[^,]", "", labs[,2]))), labs[,2], srt=90, adj=-0.1, cex=0.8)

text(labs[,1]+0.25, 0.15, labs[,2], srt=90, adj=-0.1, cex=0.8)
segments(labs[,1]+0.25, 0, labs[,1]+0.25,  0.15, col="red")

text(3.5, 1.5, "Not shown:\nLCT(x=28.7)", adj=c(0,1))
dev.off()

tiff("plots/scan_sds2.tiff", units="mm", width=25, height=25, res=300, compression="lzw")
par(list(mar=c(2.4, 2.4, 0.4, 0.1), cex=0.25))
plot(density(sdsdist$x), col="darkblue", lwd=1, ylim=c(0,1.5), bty="n", xlab="", ylab="", main="")
hist(combine$sds, add=TRUE, col=alpha("red", 0.5), border=alpha("red", 0.5), breaks=100, freq=FALSE)
dev.off()

anc <- read.table("data/Supplementary_data_table_3.txt.gz", as.is=TRUE, header=TRUE)
ancxdata <- merge(data, anc,  by=c("CHR", "POS", "ID"))
ancxdata$ANCSTAT <- qchisq(ancxdata$corrected.p, df=4, lower.tail=FALSE)
combine$anc <- NA
for(i in 1:NROW(combine)){
    tx <- ancxdata[ancxdata$CHR==combine[i,"CHR"] & abs(ancxdata$POS-combine[i,"POS"])<500000,"ANCSTAT"]
    combine[i,"anc"] <- mean(tx)
}
reg <- paste0(ancxdata$CHR, "_", floor(ancxdata$POS/1e6))
ancdist <- aggregate(ancxdata$ANCSTAT, mean, by=list(reg=reg))

cc <- combine[combine$anc>0 & combine$anc<15,]
rndanc <- floor(2*cc$anc)/2
labs<-aggregate(cc$neargene, by=list(rndanc), paste0, collapse=",")
labs <- labs[labs$x!="LINC00955",]

tiff("plots/scan_anc.tiff", units="mm", width=57, height=57, res=300, compression="lzw")
par(list(mar=c(4.4, 4.4, 0.4, 0.1), cex=0.5))
plot(density(ancdist$x), col="darkblue", lwd=3,  bty="n", xlab=expression("Mean test statistic"), main="", xlim=c(0,15))
## hist(combine$anc, add=TRUE, col=alpha("red", 0.5), breaks=seq(1, 50,0.5), freq=FALSE)
## text(labs[,1]+0.25, 0.13*(1+nchar(gsub("[^,]", "", labs[,2]))), labs[,2], srt=90, adj=-0.1, cex=0.8)

text(labs[,1]+0.25, 0.155, labs[,2], srt=90, adj=-0., cex=0.8)
segments(labs[,1]+0.25, 0, labs[,1]+0.25,  0.15, col="red")

text(10, 0.6, "Not shown:\nLCT(x=41.8)", adj=c(0,1))
dev.off()

#TODO: add lead snp info, make trajectories, rerun polygenic scan. 

output<-combine[,c( "neargene", "CHR", "start", "end", "lead.snp", "lead.alt", "lead.s")]
print(xtable(output, type = "latex"), file = "plots/sig_output.tex")

#hla.bin <- data.bin[data.bin$CHR==6 & data.bin$start>28477797 & data.bin$end<33448354,]
hla.bin <- data.bin[data.bin$CHR==6 & data.bin$start>28000000 & data.bin$end<34000000,]

pdf("plots/hla_scan.pdf",  width=6.7, height=2.2)
par(list(mar=c(4.5, 5.4, 0.4, 0.5), cex=0.5))
plot(hla.bin$POS/1e6, -log10(hla.bin$P1), col="white", xlab="Chromosome 6 position hg19 (Mb)",  ylab=expression(-log[10](P-value)), bty="n", tck = 0.01, xaxt="n", ylim=c(0,9), cex.lab=1.5)
segments(28, -0.3, 34, -0.3)
segments(28.477797, -0.3, 33.448354, -0.3, lwd=4)
segments(seq(28,34,1), -0.3, seq(28,34,1), 0)
segments(seq(280,340,1)/10, -0.3, seq(280,340,1)/10, -0.15)
mtext(c(28:34), side=1, at=c(28:34), line=0.5, cex=0.5)
rect(31.026009, 0, 31.361897, -log10(4.110306e-09), border="red" )
rect(32.083175, 0, 32.177900, -log10(6.779259e-09), border="red" )
rect(28.234597, 0, 28.374902, -log10(4.337477e-08), border="red" )

segments(hla.bin$start/1e6, -log10(hla.bin$P1), hla.bin$end/1e6, -log10(hla.bin$P1), col="darkblue", lwd=2)
abline(h=7, col="red", lty=2 )
text(c(31.026009,32.083175,28.234597), c(-log10(4.110306e-09),-log10(6.779259e-09), -log10(4.337477e-08)), c("HLA2", "HLA3", "HLA1"), adj=c(0,-0.2))
dev.off()
