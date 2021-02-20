a = "111010"
b = "0101001"
n = max(len(a), len(b))
a,b = a.zfill(n), b.zfill(n)
carry = 0
answ = []

for i in range(n-1,-1,-1):
	# print(a[i],b[i], answ[i])
	if a[i] == '1':
		carry += 1
	if b[i] == '1':
		carry += 1
	
	if carry%2==1:
		answ.append('1')
	else:
		answ.append('0')
	carry = carry//2
	print(a[i],b[i], answ, carry)
if carry==1:
	answ.append('1')
answ.reverse()

print( ''.join(answ))