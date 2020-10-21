# Scansion
### Summary
`Scansion.jl` is a collection of functions that take a string input (e.g. "Arma virumque canō, Trōiae quī prīmus ab ōrīs", _Aeneid_ 1.1) and turn it into the correct series of long and short syllables (stored as a vector e.g. [1,0,0,1,0,0,1,1,1,1,1,0,0,1,1]) that represent our modern day recreation of sounds the poet would have intended in the oral tradition, according to standard [scansion rules](http://www.thelatinlibrary.com/satire/scansion.pdf).

The goal of doing this is to eventually be able to quickly access certain summary statistics of the text (e.g. how many lines are heavily spondaic, how many are more dactylic, how many are [hypermetric](/see_all_aeneid_in_one_page/twelvebooks.png) etc. and whether they are clumped together) or if other patterns emerge.

The data (taken from [The Latin Library](https://www.thelatinlibrary.com/vergil/aen1.shtml)) are Public Domain.

## What this does so far
* The lines can be scanned with about 94% accuracy. (The errors are due to metrical anomalies such as unfinished lines, half lines or hypermetric lines, as well as possible typos or format errors with reading the text files)
* This can be done pretty fast, and on different authors (developed on Vergil, verified on Ovid with better performance!)

## What I am now working on
* Cleaning up the code to make it easier to extract the meaningful statistics mentioned above
* Researching methods to answer the "clumpiness" question (e.g. some code in `main.jl` naively tries to apply a variational auto encoder to classify rolling sections of 5 lines as dactylic or spondaic, but this makes almost no sense)
* Learning more web programming so more people can use this tool (like on http://logical.ai/arma/) and quickly arrive at hypermetric lines for example

## Credits
I was inspired by http://logical.ai/arma/ for the guessing portion of the scanner algorithm.

Thank you to Professor Stephen Rupp for supervising my project!
