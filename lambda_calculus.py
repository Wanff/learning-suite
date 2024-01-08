#%%
def eb(bool): #eb stands for evalBool
    return bool(True)(False)

true = lambda x : lambda y : x

false = lambda x : lambda y : y

#%%

# negation = lambda bool: lambda x : lambda y : bool(y)(x)

negation = lambda x : x(false)(true)

assert eb(negation(true)) == eb(false)
assert eb(negation(false)) == eb(true)
# %%
orOp = lambda x : lambda y : x(true)(y)

assert eb(orOp(false)(false)) == eb(false)
assert eb(orOp(true)(false)) == eb(true)
assert eb(orOp(false)(true)) == eb(true)
assert eb(orOp(true)(true)) == eb(true)
# %%
andOp = lambda x : lambda y : x(y)(false)

assert eb(andOp(true)(false)) == eb(false)
assert eb(andOp(false)(true)) == eb(false)
assert eb(andOp(true)(true)) == eb(true)
# %%
ifte = lambda bool : lambda x : lambda y : bool(x)(y)

assert ifte(true)("boom")("nailed it") == "boom"
assert ifte(false)("boom")("nailed it") == "nailed it"
# %%

mkPair = lambda x: lambda y : lambda bool : bool(x)(y)
    
fst = lambda pair : pair(true)

snd = lambda pair : pair(false)

assert fst(mkPair(1)("val")) == 1
assert snd(mkPair("stocks")(100)) == 100
assert fst(snd(mkPair(1)(mkPair(3)(4)))) == 3

#%%
swap = lambda pair : lambda bool : bool(pair(false))(pair(true))

assert fst(swap(mkPair("lol")("good stuff"))) == "good stuff"
assert snd(swap(swap(mkPair(1)(2)))) == 2

#%%
def en(num): #en for evalNumber
    return num(lambda x : x + 1)(0)

# Need to define succ for this one.
succ = lambda num : lambda f : lambda x : f(num(f)(x))

def toLNum(num):
    return succ(toLNum(num-1)) if num else zero

zero = lambda f : lambda x : x

num1 = toLNum(6)
num2 = toLNum(5)
two = toLNum(2)
three = toLNum(3)
four = toLNum(4)

assert en(succ(succ(succ(zero)))) == 3
# %%
add = lambda num1 : lambda num2 : num1(succ)(num2) #hmm
"""
succ is repeated and called on itself num1 N times so
it looks like:
f f f f succ (num2 == f f f f x)
and then num2 replaces succ so it becomes
f f f f f f f f x
"""

# add = lambda num1 : lambda num2: lambda f : lambda x : num1(succ(num)(f)(x))(num2(f)(x))
assert en(add(num1)(num2)) == 11
# %%
mult = lambda num1 : lambda num2 : num1(num2(succ))(zero)
mult = lambda num1 : lambda num2 : num1(add(num2))(zero)

#you add num2 amount of times num1 times to zero
assert en(mult(num1)(num2)) == 30
# %%
isZero = lambda num : 

one = lambda f : lambda x : f(x)

assert eb(isZero(zero))
assert eb(negation(isZero(one)))

# %%
pred = 

assert en(pred(three)) == 2
assert en(pred(zero)) == 0
assert en(pred(num2)) == 244