from zhipuai import ZhipuAI
client = ZhipuAI(api_key="058cd7709affe668afdcf4d71bceb31a.bMTgK0Eg4R46B5Jl")

response = client.images.generations(
    model="cogView-4", #填写需要调用的模型编码
    prompt="在干燥的沙漠环境中，一棵孤独的仙人掌在夕阳的余晖中显得格外醒目。这幅油画捕捉了仙人掌坚韧的生命力和沙漠中的壮丽景色，色彩饱满且表现力强烈。",
    size="1440x720"
)
print(response.data[0].url)