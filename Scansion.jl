module scansion

#=testing
x1="Interea extremo bellator in aequore Turnus"
x2="Arma virumque cano Troiae qui primus ab oris"
x3="Italiam fato profugus Lavinaque venit"
x4= "Albanique patres atque altae moenia Romae"
x= "cui pater intactam dederat, primisque iugarat"
x="coniugis, ora modis attollens pallida miris"
x="quippe domum timet ambiguam Tyriosque bilinguis"
lines = [x1,x2,x3,x4]
test_el = turn_line_into_syllables(parse_uv_ij(x))
test_knownq = make_elisions_and_center(test_el)
test_guesser = known_quantities(test_knownq)
add_n_dactyls(test_guesser,length(test_guesser)-12)
scan(test_guesser)
scan_lines(lines)
=#

#=reference
vowels = ["a", "e", "i","o","u"]
diphthongs = ["ae", "au", "ei", "eu", "oe"]
long = 2
short = 1
=#


function parse_uv_ij(x)
    for i in 1:length(x)-1
        if (i == 1) && occursin(r"[aeiou]"i ,x[i+1:i+1]) && occursin(r"i"i, x[i:i])
            print((i == 1) && occursin(r"[aeiou]"i ,x[i+1:i+1]) && occursin(r"i"i, x[i:i]))
            x = replace(x, r"[i]"i => s"j",count = 1)
        else
            x = replace(x, r"([aeiou\s])(i)([aeiou])"i => s"\g<1>j\g<3>")
        end
    end
    return x
end

function turn_line_into_syllables(x)
    lst = Any[]
    start = 1
    while length(x) != 0
        m = match(r"(qu|ch|ph|th|[^a^e^i^o^u]*)*(ae|au|ei|eu|oe|[aeiou])(([^aeiou]*qu)|[^aeiou]ch|[^aeiou]ph|[^aeiou]th|[^aeiou\s]*)\s?"i,x) ## TODO:SEE IF THIS CAN BE SIMPLIFIED
        push!(lst,m.match)
        x = x[length(m.match)+1:end]
    end
    return lst
end

function make_elisions_and_center(syll)
    ##Make elisions: delete the elided syllable for forward elision, TODO: mash the prodelisions
    for i in 1:(length(syll)-1)
        if syll[i][end] == ' '
            m = match(r"ae|au|ei|eu|oe|[aeioum]\s$"i,syll[i])
            n = match(r"^[haeiou]"i,syll[i+1])
            if (m != nothing)&&(n != nothing)
                syll[i] = " "
            end
        end
    end

    f(x)=(x != " ")
    newsyll = filter(f, syll)

    ##Make sure each syllable starts with a vowel
    for i in 2:(length(newsyll)-1)
        m = match(r"^[^aeiou]+"i , newsyll[i]) ##find consonants at beginning of syllable
        if m !== nothing
            newsyll[i] = newsyll[i][length(m.match)+1:end]
            newsyll[i-1] = join([newsyll[i-1], m.match])
        end
    end
    return newsyll
end

## all hard facts implemented
##guessing begins below

function known_quantities(syll1; guess_o = true, guess_es = true)
    vec = zeros(length(syll1))
    vec[end]=2 #for anceps
    vec[end-1] = 2 #for final foot first syllable
    vec[1] = 2 #for first long

    ##long by position and diphthongs and guess o
    for i in 1:length(syll1)
        m = match(r"[^aeiouh]+"i,syll1[i])
        if m !== nothing
            if ((occursin(r"[^aeiouh\s]\s*[^aeiouh\s]"i,m.match)) && !(occursin(r"[bcdfgpt][lmnr]"i,m.match))) | occursin(r"x|z"i, m.match)
                vec[i] = 2
            end
        end
        v = match(r"ae|au|ei|eu|oe"i,syll1[i])
        if v !== nothing
            vec[i] = 2
        end
        if guess_o == true
            om = match(r"o\s"i, syll1[i])
            if om !== nothing
                vec[i] = 2
            end
        end
        if guess_es == true
            esm = match(r"es\s"i, syll1[i])
            if esm !==nothing
                vec[i] = 2
            end
        end
    end
    return vec
end


function add_n_dactyls(vec,n; start = length(vec)-2)
    ##end of line priority
    count = 0
    for i in start:-1:3 ##check the spots that can take a dactyl
        if count < n && ((vec[i] == 0 && vec[i-1] == 0 && vec[i-2] == 0) | (vec[i] == 0 && vec[i-1] == 0 && vec[i-2] == 2))
            vec[i] = 1
            vec[i-1] = 1
            vec[i-2] = 2
            count = count+1
        end
    end
    return (vec, count==(n))
end

function make_all_else_long(vec)
    for i in 1:length(vec)
        if vec[i] == 0
            vec[i] = 2
        end
    end
    return vec
end

##up to here
##the strings are now numeric



function it_scans(vec)
    return sum(vec)==24
end


function scan(vec) #takes numbers
    num_dactyls = length(vec)-12 #http://logical.ai/arma/\
    vecop = add_n_dactyls(vec,num_dactyls)
    start = length(vec)-4
    if vecop[2] == true
        vec = make_all_else_long(vecop[1])
    else
        vec = add_n_dactyls(vec, num_dactyls, start= start)
        vec = make_all_else_long(vec[1])
    end
    return (vec, it_scans(vec))
end


function scan_line(x) #takes cleanish string
    return scan(known_quantities(make_elisions_and_center(turn_line_into_syllables(parse_uv_ij(x)))))
end

function scan_lines(lines)
    scan = Any[]
    special_lines = Any[]
    for i in 1:length(lines)
        nums = scan_line(lines[i])
        if nums[2]
            push!(scan, nums[1])
        else
            push!(special_lines, (lines[i], i))
        end
    end
    return (scan, special_lines)
end


testing
book2 = readlines("Ovid met book1.txt")
function clean_data(lines)
    for i in 1:length(lines)
        lines[i] = replace(lines[i], r"[ā]"i => s"a")
        lines[i] = replace(lines[i], r"[ē]"i => s"e")
        lines[i] = replace(lines[i], r"[ī]"i => s"i")
        lines[i] = replace(lines[i], r"[ō]"i => s"o")
        lines[i] = replace(lines[i], r"[ū]"i => s"u")
        lines[i] = replace(lines[i], r"[^a-zA-Z\s]" => s"")
        lines[i] = rstrip(lines[i])
        lines[i] = lstrip(lines[i])
    end
    f(x)=(x != "")
    lines = filter(f, lines)

    return lines
end
clean_book2 = clean_data(book2)
diditscan = []
for l in clean_book2
    push!(diditscan, scan_line(l)[2])
    print(scan_line(l)[2])
end
sum(diditscan)
