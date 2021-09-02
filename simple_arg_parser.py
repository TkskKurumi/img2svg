def parse_args(s,flag_start='-',default_flag='default'):
	in_quote=False
	flag=default_flag
	in_flag=False
	ret={}
	content=''
	prev=' '
	
	for i in s:
		
			
		if(in_quote):
			if(i=='"'):
				in_quote=False
			else:
				content+=i
				ret[flag]=content
		elif(in_flag):
			if(i==' '):
				in_flag=False
				ret[flag]=True
				content=''
			else:
				flag+=i
		else:
			if(i=='"'):
				in_quote=True
				content=''
			elif(i==flag_start and prev==' '):
				flag=''
				in_flag=True
			else:
				content+=i
				ret[flag]=content
		prev=i
	if(in_flag):
		in_flag=False
		ret[flag]=True
	return ret
if(__name__=='__main__'):
	dic=parse_args(r'  千千 -suffix \"哒！\\ -u 402254524 enmiao az -no_lines -nd')
	print(dic)
	print(dic.get("dne"))