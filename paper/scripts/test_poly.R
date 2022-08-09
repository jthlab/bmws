library(ggplot2)
library(ggrepel)
library(xtable)

traits <- c("SkinCol_UKB", "Height_UKB", "SitHeight_UKB", "Height_GIANT",  "BMI_UKB", "BMD_UKB", "BMR_UKB", "Weight_UKB", "FM_UKB", "FFM_UKB", "Waist_UKB", "Hip_UKB",  "WBC_UKB","RBC_UKB","Pulse_UKB","HDL_UKB","LDL_UKB","TG_UKB", "AAM_UKB", "BW_EGG", "HC_EGG", "VitD_UKB", "Calcium_UKB", "Folate_UKB", "FI_MAGIC", "FG_MAGIC", "2hGlu_MAGIC", "HbA1c_MAGIC")

#"EA_UKB"
# "NEB", 
#"UKB_WBsib_Height" "UKB_WBsib_BMD",
#"BP_SYS_UKB","BP_DIA_UKB",

p.cutoffs <- c(4,6,8)

blank <- rep(NA, length(traits))

selstats <- read.table("../data/s_scan_all_brit.txt.gz", as.is=TRUE, header=FALSE)
colnames(selstats) <- c("CHR", "POS", "MarkerName", "REF", "ALT", "FREQ", "S", "S1", "S2")

results <- data.frame(trait=traits)
for(pc in p.cutoffs){
    results[,paste0("cor.", pc)] <- NA
    results[,paste0("p.", pc)] <- NA
    results[,paste0("N.", pc)] <- NA
}

for(what in traits){
    for(pc in p.cutoffs){
        pcc <- 10^(-pc)
        gwas <- paste0("../data/gwas_results/", what, "_250kb_Capture_intersection.txt")
        gwas <- read.table(gwas, as.is=TRUE, header=TRUE)
        data <- merge(gwas, selstats, by=c("CHR", "POS", "MarkerName", "REF", "ALT"))
        data <- data[data$FREQ>0.05 & data$FREQ<0.95,]
        data <- data[data$CHR!=6 | data$POS<27e6 | data$POS>32e6,] #remove HLA
        data <- data[data$p<pcc,]
        if(NROW(data)>2){
            results[results$trait==what,paste0("N.", pc)] <- NROW(data)
            cc<-cor.test(data$ALT.EF, data$S)
            results[results$trait==what,paste0("p.", pc)] <- cc$p.value
            results[results$trait==what,paste0("cor.", pc)] <- -cc$estimate
        }
    }
}

for(pv in c(4,6,8)){
pdf(paste0("poly_scan_", pv, ".pdf"))
dd <- results[,c("trait", paste0(c("cor.", "p."), pv))]
colnames(dd) <- c("trait", "cor", "logp")
dd$logp <- -log10(dd$logp)
p <- ggplot(data=dd, aes(x=cor, y=logp, label=trait))+geom_point(col="darkblue")+geom_text_repel(size=2,max.overlaps = Inf)
p <- p+xlab(expression(Correlation~between ~ beta ~ and ~ bar(s)))+ylab(expression(-log[10]~(P-value)))
p <- p+geom_hline(yintercept=-log10(0.05/NROW(results)), lty=2, col="red")
p <- p+theme_light()
print(p)
dev.off()
}

rr<-results[,c("trait", "cor.4", "p.4", "N.4")]
rr$ref<-NA
rr$desc<-NA
rr<-rr[,c("trait", "desc", "ref", "cor.4", "p.4", "N.4")]
print(xtable(rr, include.rownames=FALSE, include.colnames=TRUE), file="poly.tex")
