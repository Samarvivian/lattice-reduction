q =122430513841
f=231231
g=195698
inv_f=pow(f,-1,q)  # 计算逆元
h=inv_f*g%q
m=123456
r=101010
e=(r*h+m)%q
a=f*e%q
#print(a)
inv_f=pow(f,-1,g)
b=inv_f*a%g
print(b)