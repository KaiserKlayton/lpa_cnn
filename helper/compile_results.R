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
models <- models[sapply(models, function(x) { !file.exists(paste0("results/", x, "_accuracies.csv")) })]

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
        if (length(table(md$score)) == 2)
            result <- round(table(md$score)[2] / nrow(md), 4) * 100
        else if (0 %in% md$score) {
            result <- 0
        }
        else {
            result <- 100
        }

        return(result)
    })

    accuracies.top.5 <- lapply(mds, function(md) {
        if (length(table(md$score.top.5)) == 2)
            result <- round(table(md$score.top.5)[2] / nrow(md), 4) * 100
        else if (0 %in% md$score.top.5) {
            result <- 0
        } else {
            result <- 100
        }

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
speed.files <- list.files("results", pattern = "*.txt", full.names = TRUE)

for (f in speed.files) {
    model.name <- str_match(f, "results/run_log_(.+)_.+[.]txt")[2]
    mode.name <- str_match(f, "results/run_log_.+_(.+)[.]txt")[2]

    if (file.exists(paste0(file.path("results", paste(model.name, mode.name, "speeds.csv", sep = "_"))))) {
        next
    }

    temp.data <- read.csv(f, header = FALSE, sep = ":", stringsAsFactors = FALSE)

    if (mode.name == "eigen") {
        data <- temp.data[1:(nrow(temp.data) - 2), ]
        ids <- data$V2[data$V1 == "image"]
        gaps <- as.numeric(row.names(data)[data$V1 == "image"])
        gap <- diff(gaps)
        data$V2 <- as.integer(rep(ids, each = gap))
        data <- data[data$V1 != "image", ]
        chunk.size <- nrow(data[data$V2 == 0, ]) / 2
        temp <- vector(length = nrow(data))
        temp <- rep(c(1:chunk.size), each = 2)
        data$V2 <- temp
        data <- cbind(data[c(TRUE, FALSE), ],
                      data[c(FALSE, TRUE), ])[, 1:3]
        data[, 3] <- str_replace(data[, 3], ".0$", "")
        data$V1 <- paste0("(", data$V2, ") ", data$V1)
        data <- data[, c(1,3)]
    } else {
        data <- temp.data
        data <- temp.data[1:(nrow(temp.data) - 2), ]
        ids <- data$V2[data$V1 == "image"]
        sizes <- data$V1[is.na(data$V2)]
        gaps <- as.numeric(row.names(data)[data$V1 == "image"])
        gap <- diff(gaps)
        data$V3 <- as.integer(rep(ids, each = gap))
        data <- data[data$V1 != "image", ]
        chunk.size <- nrow(data[data$V3 == 0, ]) / 8
        temp <- vector(length = nrow(data))
        temp <- rep(c(1:chunk.size), each = 8)
        data$V3 <- temp
        data$V4 <- rep(sizes, each = 8)
        data$V2 <- str_replace(data$V2, ".0$", "")
        data$V1 <- paste(paste0("(", data$V3, ")"), data$V4, data$V1)
        data <- data[complete.cases(data), ]
        data <- data[, 1:2]
    }

    names(data) <- c("V1", "V2")
    data <- rbind(data, temp.data[(nrow(temp.data) - 1):nrow(temp.data), ])
    data$V2 <- as.numeric(data$V2)
    row.names(data) <- c(1:nrow(data))
    targets <- unique(data$V1)
    totals <- aggregate(V2 ~ V1, data = data, sum)
    avgs <- aggregate(V2 ~ V1, data = data, mean)
    data <- merge(avgs, totals, by = "V1")
    data <- data[match(targets, data$V1), ]
    row.names(data) <- c(1:nrow(data))
    names(data) <- c("measure", "batch avg (ms)", "batch total (ms)")
    data$measure[c(nrow(data) - 1, nrow(data))] <- c("online run time", "aggregate GEMM time")
    data$`batch avg (ms)` <- round(data$`batch avg (ms)` * 1000, 4)
    data$`batch total (ms)` <- round(data$`batch total (ms)` * 1000, 4)
    data[(nrow(data) - 1):nrow(data), ]$`batch total (ms)` <- round(data[(nrow(data) - 1):nrow(data), ]$`batch total (ms)` * 1000, 4)

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
