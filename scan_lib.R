#Return CHR, POS, SL1, SL2
window.scan <- function(data, bin.size=20, bin.skip=10)
{
    chrs <- unique(data$CHR)
    windows <- list()
    chr.i=1
    for(chr in chrs){
        this.data <- data[data$CHR==chr,]
        n.windows <- round((NROW(this.data)-bin.size)/bin.skip)+1
        empty <- rep(NA, n.windows)
        this.win <- data.frame(CHR=rep(chr,n.windows), POS=empty,
                               start=empty, end=empty,
                               SL1=empty, SL2=empty,
                               lead.snp=rep("",n.windows),
                               lead.alt=rep("",n.windows),
                               lead.pos=rep(0,n.windows),
                               lead.s=rep(0,n.windows))

        for( i in 1:n.windows){
            start=1+(i-1)*bin.skip
            end=start+bin.size-1
            if(end>NROW(this.data)){ #last bin
                end <- NROW(this.data)
                start <- end-bin.size+1
            }
            win.data <- this.data[start:end,]
            this.win[i,"POS"] <- mean(win.data$POS)
            this.win[i,"start"] <- min(win.data$POS)
            this.win[i,"end"] <- max(win.data$POS)
            this.win[i,"SL1"] <- mean(win.data$SL1)
            this.win[i,"SL2"] <- mean(win.data$SL2)

            this.lead <- which.max(win.data$SL1)[1]
            
            this.win[i,"lead.snp"] <- win.data[this.lead, "ID"]
            this.win[i,"lead.alt"] <- win.data[this.lead, "ALT"]
            this.win[i,"lead.s"] <- win.data[this.lead, "S.HAT"]
            this.win[i,"lead.pos"] <- win.data[this.lead, "POS"]
        }
        windows[[chr.i]] <- this.win
        chr.i <- chr.i+1
    }
    windows <- do.call(rbind, windows)
    return(windows)
}
