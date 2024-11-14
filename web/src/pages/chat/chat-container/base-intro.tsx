import { Divider, Typography } from 'antd';

const { Title, Paragraph } = Typography;

const RenderIntro = ({ selectedValue ,language}) => {
  if (selectedValue === '程序开发' ) {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>程序开发提示词示例：</Title>
        <Divider />
        <Paragraph>在Linux环境中，如何统计当前目录下文件夹的大小？</Paragraph>
        <Paragraph>
          如何使用Ansible playbook 或如何配置 Linux服务器 yum 源？
        </Paragraph>
        <Paragraph>Python中如何定义一个函数？</Paragraph>
        <Paragraph>SQL语句中如何进行联合查询？</Paragraph>
      </div>
    );
  }else if (selectedValue === 'CODE'){
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>Sample Prompts for Program Development:</Title>
        <Divider />
        <Paragraph>How can I calculate the size of folders in the current directory in a Linux environment?</Paragraph>
        <Paragraph>
          How do I use an Ansible playbook, or how do I configure the yum repository on a Linux server?
        </Paragraph>
        <Paragraph>How do you define a function in Python?</Paragraph>
        <Paragraph>How do you perform a join query in SQL?</Paragraph>
      </div>

    );
  }
   else if (selectedValue == '自由对话') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>自由对话提示词示例：</Title>
        <Divider />
        <Paragraph>如何进行数据库的迁移，迁移时要注意哪些风险？</Paragraph>
        <Paragraph>什么是大语言模型提示词，其作用是什么？</Paragraph>
        <Paragraph>
          请介绍一下ARM服务器和x86服务器，并对两者进行比较。
        </Paragraph>
      </div>
    );
  } else if (selectedValue == 'CHAT') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>Sample Prompts for Open Conversation:</Title>
        <Divider />
        <Paragraph>How do you perform database migration, and what risks should be considered during migration?</Paragraph>
        <Paragraph>What is a large language model prompt, and what is its purpose?</Paragraph>
        <Paragraph>
          Please introduce ARM servers and x86 servers and provide a comparison between the two.
        </Paragraph>
      </div>
    );
  } else if (selectedValue == '知识库') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>知识库提示词示例：</Title>
        <Divider />
        <Paragraph>
          在大型模型和专业知识库的支持下，可以为您提供更专业的的IT运维支持和解决方案。
        </Paragraph>
        <Paragraph>
          请您选择相应领域的知识库，我将基于知识库的内容，为您提供更有针对性的建议和解决方案。
        </Paragraph>
        <Paragraph>
          比如有关达梦数据库，华为泰山服务器和Ansible自动化方面的支持。
        </Paragraph>
      </div>
    );
  } else if (selectedValue == 'KNOWLEDGE') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>Sample Prompts for Knowledge Base:</Title>
        <Divider />
        <Paragraph>
          With the support of large models and a professional knowledge base, I can provide you with more specialized IT operations support and solutions.
        </Paragraph>
        <Paragraph>
          Please select a relevant knowledge base from your field, and I will offer more targeted advice and solutions based on its content.
        </Paragraph>
        <Paragraph>
          For example, support related to Dameng databases, Huawei TaiShan servers, and Ansible automation.
        </Paragraph>
      </div>
    );
  } else if (selectedValue == '日志分析') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>日志分析提示词示例：</Title>
        <Divider />
        <Paragraph>
          请上传相关日志文件，我将解读日志内容，分析潜在问题，并尽力给出根本原因，解决方案和预防建议。
        </Paragraph>
      </div>
    );
  }else if (selectedValue == 'LOG') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>Sample Prompts for Log Analysis:</Title>
        <Divider />
        <Paragraph>
          Please upload the relevant log files. I will interpret the log contents, analyze potential issues, and do my best to provide root causes, solutions, and preventive recommendations.
        </Paragraph>
      </div>
    );
  } else if (selectedValue == 'CMDB') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>CMDB 提示词示例：</Title>
        <Divider />
        <Paragraph>您可以问“北湖数据中心目前有多少设备”</Paragraph>
        <Paragraph>您可以问“ 按设备类型分组，统计每种设备类型的数量”</Paragraph>
        <Paragraph>
          您可以问“查询并统计设备类型为小型机在数据中心的分布情况”
        </Paragraph>
        <Paragraph>
          您可以问“查询并统计设备类型为ARM服务器采购时间的分布，以年为单位统计“
        </Paragraph>
      </div>
    );
  } else {
    return (language === 'en' ? (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>Hi! I’m IntelliOps, your all-in-one IT operations assistant.</Title>
        <Divider />
        <Paragraph>I have six core capabilities, always ready to support you:</Paragraph>
        <Paragraph>
          <b>CHAT</b>: Whether you have everyday questions or IT-related issues, feel free to chat with me. I’m here to provide answers.
        </Paragraph>
        <Paragraph>
        <b>CODE</b>: Powered by a large model, I’m focused on helping you resolve any programming challenges. From programming languages and algorithms to development tools, I’ll do my best to offer accurate and detailed help and guidance.
        </Paragraph>
        <Paragraph>
        <b>CMDB </b>: Through natural language conversation, I can assist you in managing CMDB assets effortlessly, keeping your system organized at all times.
        </Paragraph>
        <Paragraph>
        <b> Log Analysis</b>: Upload your log files, and I’ll analyze the contents, identify potential issues, and provide root cause analysis, solutions, and prevention advice.
        </Paragraph>
        <Paragraph>
        <b> Knowledge Base</b>: Supported by a large model and professional knowledge base, I provide specialized IT operations support and solutions.
        </Paragraph>
        <Paragraph>
        <b> Automated Operations</b>: I excel in automated operations, especially with Ansible. I can interact with real external environments and engage in ongoing, contextual exchanges. If human approval is required, I can facilitate that process as well. You can ask me to “list all skills in the skill library,” and I’ll display the relevant skills for you.
        </Paragraph>
        <Paragraph>
          No matter what challenges you encounter, feel free to reach out. I’m here to provide you with top-quality service!
        </Paragraph>
      </div>
      ):(
        <div style={{ margin: '60px 20px', color: 'grey' }}>
          <Title level={4}>嗨！我是小吉，您的全能IT运维助手.</Title>
          <Divider />
          <Paragraph>我拥有六大核心能力，随时为您提供支持：</Paragraph>
          <Paragraph>
            自由对话：无论是日常问题，还是其他IT相关问题，都可以与我畅谈，我将为您提供解答。
          </Paragraph>
          <Paragraph>
            程序开发：基于大型模型构建的，专注于帮助您解决程序开发中遇到的各种问题。无论是关于编程语言、算法还是开发工具，我都会尽力为您提供准确、详细的帮助和指导。
          </Paragraph>
          <Paragraph>
            CMDB管理：通过自然语言对话，帮助您轻松管理CMDB资产，确保您的系统始终井然有序。
          </Paragraph>
          <Paragraph>
            日志分析：上传日志文件，我会帮您解析日志内容，识别潜在问题，并提供根本原因分析、解决方案以及预防建议。
          </Paragraph>
          <Paragraph>
            知识库问答：凭借大型模型和专业知识库的支持，我为您提供更专业的IT运维支持和解决方案。
          </Paragraph>
          <Paragraph>
            自动化运维：我擅长自动化运维，尤其在使用Ansible方面。我能够与外部真实环境互动，并根据上下文进行有反馈的持续交流。如果需要人工介入审批，我也能支持此流程。您可以问我“列出所有技能库中的技能”，我会为您展示相关的技能。
          </Paragraph>
          <Paragraph>
            无论您遇到什么挑战，都可以随时向我求助。我将竭诚为您提供最优质的服务！
          </Paragraph>
        </div>
        )
    );
  }
};

export default RenderIntro;
