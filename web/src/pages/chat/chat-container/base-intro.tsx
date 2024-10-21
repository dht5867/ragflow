import { Divider, Typography } from 'antd';

const { Title, Paragraph } = Typography;

const RenderIntro = ({ selectedValue }) => {
  if (selectedValue === '程序开发') {
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
  } else if (selectedValue == '自由对话') {
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
  } else if (selectedValue == '自动化运维') {
    return (
      <div style={{ margin: '60px 20px', color: 'grey' }}>
        <Title level={4}>自动化运维 提示词示例：</Title>
        <Divider />
        <Paragraph>请列出所有技能库中的技能</Paragraph>
      </div>
    );
  } else {
    return (
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
    );
  }
};

export default RenderIntro;
