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

        label <- scan(pipe(paste("cut -f1 -d,", input.file)))

        # combine.
        temp.2 <-  cbind(predictions, label)

        return(temp.2)
    }))

    # score.
    model.results$score <- apply(model.results, 1, function(x) {
        preds <- x[1:(length(x) - 3)]
        label <- x['label']
        prediction <- names(preds)[which.max(preds)]

        if (as.integer(prediction) == as.integer(label)) {
            return(1)
        } else {
            return(0)
        }
    })

    mds <- split(model.results, model.results$mode)
    accuracies <- lapply(mds, function(md) {
        result <- round(table(md$score)[2] / 1000, 4) * 100

        return(result)
    })

    accuracies <- do.call(rbind, accuracies)
    accuracies <- as.data.frame(cbind(row.names(accuracies), accuracies))
    names(accuracies) <- c("mode", "classification_accuracy (%)")
    accuracies$classification_error <- round(100 - as.numeric(as.character(accuracies$`classification_accuracy (%)`)), 4)

    write.table(accuracies, paste0("results/", model, "_accuracies.csv"), row.names = FALSE, sep = ",")
}

############
## SPEEDS ##
############
speed.files <- list.files("results", pattern = "*.txt", full.names = TRUE)
for (f in speed.files) {
    model.name <- str_match(f, "results/run_log_(.+)_.+[.]txt")[2]
    mode.name <- str_match(f, "results/run_log_.+_(.+)[.]txt")[2]

    speed.data <- read.csv(f, header = FALSE, sep = ":")
    speed.data <- speed.data[!speed.data$V1 == "image", ]

    if (mode.name == "gemmlowp") {
        totals <- aggregate(V2 ~ V1, data = speed.data, sum)
        avgs <- aggregate(V2 ~ V1, data = speed.data, mean)

        data <- merge(avgs, totals, by = "V1")
        names(data) <- c("stage", "batch_avg (ms)", "batch_total (ms)")
        data$`batch_avg (ms)` <- round(data$`batch_avg (ms)` * 1000, 3)
        data$`batch_total (ms)` <- data$`batch_total (ms)` * 1000

    } else {
        data <- speed.data
        names(data) <- c("measure", "batch_avg (ms)")
        data$`batch_total (ms)` <- data$`batch_avg (ms)` * 1000
        data$`batch_avg (ms)` <- round(data$`batch_avg (ms)` * 1000, 3)
        data$`batch_total (ms)` <- data$`batch_total (ms)` * 1000
    }

    write.table(data, file.path("results", paste(model.name, mode.name, "speeds.csv", sep = "_")), row.names = FALSE, sep = ",")
}
