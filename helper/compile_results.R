#!/usr/bin/env Rscript

## Copyright: 2017
## Author: C. Clayton Violand

## Collects the following results from file and compiles into a nice table:
##
## I. Accuracies
## II. Speed
##

options(warn = -1)

library(gtools)
library(stringr)

modes <- c("caffe", "eigen", "gemmlowp")
models <- list.files("models", pattern = "[^[README.md]]*")

################
## Accuracies ##
################
# prepare table of results.
for (model in models) {  # for each model...
    model.results <- do.call(rbind, lapply(modes, function(m) {   # and for each mode...
        scores.path <- file.path("features", model, m)
        files <- list.files(scores.path, pattern = ".*[0-9].*", full.names = TRUE)
        files <- mixedsort(sort(files))

        # get the predictions.
        predictions <- do.call(rbind, lapply(files, function(file) {
            temp <- read.csv(file, header = FALSE)
            names(temp) <- c(0:(length(temp) - 1))
            # attach relevant information.
            temp$model <- model
            temp$mode <- m

            return(temp)
        }))

        # get the labels.
        input.file <- list.files(file.path("inputs", model, "production"),
                                 pattern = ".csv", full.names = TRUE)

        label <- scan(pipe(paste("cut -f1 -d,", input.file)))[1:nrow(predictions)]

        # combine.
        temp.2 <-  cbind(predictions, label)

        return(temp.2)
    }))

    # score.
    model.results$score <- apply(model.results, 1, function(x) {
        preds <- x[1:(length(x) - 3)]
        label <- x['label']
        prediction <- names(preds)[which.max(as.numeric(preds))]

        if (as.integer(prediction) == as.integer(label)) {
            return(1)
        } else {
            return(0)
        }
    })

    # score top 5.
    model.results$score.top.5 <- apply(model.results, 1, function(x) {
        preds <- as.numeric(x[1:(length(x) - 4)])
        label <- x['label']
        top.5 <- match(tail(sort(as.numeric(preds)), 5), preds) - 1
        if (label %in% top.5) {
            return(1)
        } else {
            return(0)
        }
    })

    mds <- split(model.results, model.results$mode)

    accuracies <- lapply(mds, function(md) {
        result <- round(table(md$score)[2] / nrow(md), 4) * 100

        return(result)
    })

    accuracies.top.5 <- lapply(mds, function(md) {
        result <- round(table(md$score.top.5)[2] / nrow(md), 4) * 100

        return(result)
    })

    accuracies <- do.call(rbind, accuracies)
    accuracies <- as.data.frame(cbind(row.names(accuracies), accuracies))
    names(accuracies) <- c("mode", "top-1 accuracy (%)")
    accuracies$`top-1 error` <- round(100 - as.numeric(as.character(accuracies$`top-1 accuracy (%)`)), 4)

    accuracies.top.5 <- do.call(rbind, accuracies.top.5)
    accuracies.top.5 <- as.data.frame(cbind(row.names(accuracies.top.5), accuracies.top.5))
    names(accuracies.top.5) <- c("mode", "top-5 accuracy (%)")
    accuracies.top.5$`top-5 error` <- round(100 - as.numeric(as.character(accuracies.top.5$`top-5 accuracy (%)`)), 4)

    accuracies$`top-5 accuracy (%)` <- round(as.numeric(as.character(accuracies.top.5$`top-5 accuracy (%)`)), 4)
    accuracies$`top-5 error` <- round(100 - as.numeric(as.character(accuracies.top.5$`top-5 accuracy (%)`)), 4)

    write.table(accuracies, paste0("results/", model, "_accuracies.csv"), row.names = FALSE, sep = ",")
}

############
## SPEEDS ##
############
n <- nrow(model.results) / length(models) / 3
speed.files <- list.files("results", pattern = "*.txt", full.names = TRUE)
for (f in speed.files) {
    model.name <- str_match(f, "results/run_log_(.+)_.+[.]txt")[2]
    mode.name <- str_match(f, "results/run_log_.+_(.+)[.]txt")[2]
    temp.data <- read.csv(f, header = FALSE, sep = ":", stringsAsFactors = FALSE)

    if (mode.name == "eigen") {
        data <- temp.data
        count <- 1
        for (i in 1:nrow(data)) {
            type <- str_count(data[i, 1], "[.]")
            if (type == 4) {
                data[i, 1] <- paste(paste0("(", count, ")"), data[i, 1])
                count <- count + 1
            } else if (type == 2) {
                data[i, 1] <- str_replace(data[i, 1], ".0$", "")
                data[i - 1, 2] <- as.numeric(data[i, 1])
            } else if (data[i, 1] == "image") {
                count <- 1
            }
        }
    } else {
        data <- temp.data
        count <- 1
        for (i in 1:nrow(data)) {
            type <- str_count(data[i, 1], "[.]")
            if (type == 4) {
                data[i, 1] <- paste(paste0("(", count, ")"), data[i, 1])
                dims <- data[i, 1]
                count <- count + 1
            } else if (type == 0 & !data[i, 1] %in% c("image", "average online run time", "average total time for GEMM")) {
                data[i, 1] <- paste(dims, data[i, 1])
            } else if (data[i, 1] == "image") {
                count <- 1
            }
        }
    }

    data <- data[!temp.data$V1 == "image", ]
    data <- data[complete.cases(data), ]
    totals <- aggregate(V2 ~ V1, data = data, sum)
    avgs <- aggregate(V2 ~ V1, data = data, mean)
    data <- merge(avgs, totals, by = "V1")
    names(data) <- c("measure", "batch avg (ms)", "batch total (ms)")
    row.names(data) <- c(1:nrow(data))
    data$measure[c(nrow(data) - 1, nrow(data))] <- c("online run time", "aggregate GEMM time")
    data$`batch avg (ms)` <- round(data$`batch avg (ms)` * 1000, 4)
    data$`batch total (ms)` <- round(data$`batch total (ms)` * 1000, 4)
    data[(nrow(data) - 1):nrow(data), ]$`batch total (ms)` <- round(data[(nrow(data) - 1):nrow(data), ]$`batch total (ms)` * n, 4)

    if (mode.name == "gemmlowp") {
        data.minimized <- apply(data, 2, function(x) {
            x <- str_replace(x, "[(][0-9]+[)] [0-9]+.[0-9]+.[0-9]+.[0-9]+.[0-9]+", "")
        })[-c(nrow(data) - 1, nrow(data)), ]
        totals <- aggregate(as.numeric(as.character(`batch total (ms)`)) ~ measure, data = data.minimized, sum)
        avgs <- aggregate(as.numeric(as.character(`batch avg (ms)`)) ~ measure, data = data.minimized, sum)
        extra.data <- merge(avgs, totals, by = "measure")
        extra.data$measure <- c("Aggregate convert_from_eigen time", "Aggregate convert_to_eigen time", "Aggregate dequantize time", "Aggregate gemm time", "Aggregate get_params time", "Aggregate quantize time", "Aggregate quantize_offline time")
        names(extra.data) <- c("measure", "batch avg (ms)", "batch total (ms)")
        extra.data <- extra.data[extra.data$measure != "Aggregate gemm time", ]
        data <- rbind(data, extra.data)
    }

    write.table(data, file.path("results", paste(model.name, mode.name, "speeds.csv", sep = "_")), row.names = FALSE, sep = ",")
}
