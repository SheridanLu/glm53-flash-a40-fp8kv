import json, http.client, random, re
H="/data/glm_test/fp8kv"
def chat(msgs, mt=400):
    c = http.client.HTTPConnection("127.0.0.1", 18010, timeout=600)
    c.request("POST","/v1/chat/completions", json.dumps({"model":"glm-5.3-flash","messages":msgs,"max_tokens":mt,
        "temperature":0,"chat_template_kwargs":{"reasoning_effort":"low","clear_thinking":True}}), {"Content-Type":"application/json"})
    r=c.getresponse(); d=r.read().decode("utf-8","replace"); c.close()
    j=json.loads(d); m=j["choices"][0]["message"]
    return (m.get("content") or "")
def extract(t):
    blocks=re.findall(r"```(?:python)?\n(.*?)```", t, re.S)
    return blocks[0] if blocks else t
ok=0; cases=[("add","def add(a,b)","验证 add(2,3)==5",lambda f: f(2,3)==5),
             ("square","def square(x)","验证 square(7)==49",lambda f: f(7)==49),
             ("rev","def rev(s)","验证 rev('abc')=='cba'",lambda f: f("abc")=="cba")]
for name,sig,_,check in cases:
    out=chat([{"role":"user","content":f"用 Python 写一个函数 {sig}。只给代码块,不要解释。"}])
    src=extract(out)
    try: ns={}; exec(src,ns); good=bool(check(ns[name]))
    except Exception as e: good=False; print(f"  {name}: {type(e).__name__}")
    ok+=good; print(f"{name}: {'OK' if good else 'FAIL'}")
print(f"代码(chat 接口): {ok}/3")
# 算术复测: 看错的是不是近似的
bad=[]
for i in range(12):
    rnd=random.Random(i); a,b,c=rnd.randint(11,99),rnd.randint(11,99),rnd.randint(100,999)
    out=chat([{"role":"user","content":f"Compute {a} * {b} + {c} = ? Reply with the number only."}], 200)
    m=re.search(r"-?\d+", out.replace(",",""))
    got=int(m.group()) if m else None
    if got!=a*b+c: bad.append((a,b,c,a*b+c,got))
print(f"算术(chat 接口): {12-len(bad)}/12; 错的: {bad}")
