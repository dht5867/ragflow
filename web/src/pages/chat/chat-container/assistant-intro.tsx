import { QuestionCircleOutlined } from '@ant-design/icons';
import { FloatButton, Modal, Typography } from 'antd';
import { useState } from 'react';

import 'antd/dist/reset.css'; // 引入 Ant Design 样式

const { Paragraph } = Typography;

const AssistantIntro = () => {
  const [visible, setVisible] = useState(false);

  const showModal = () => setVisible(true);
  const handleCancel = () => setVisible(false);

  return (
    <div>
      <FloatButton
        type="default"
        icon={<QuestionCircleOutlined />}
        tooltip="查看提示词示例"
        onClick={showModal}
        style={{ margin: '100px 0' }}
      />
      <Modal
        title="小吉助手提示词示例"
        open={visible}
        onCancel={handleCancel}
        footer={null}
        style={{ top: 20, color: 'grey' }}
        width={800}
      >
        <div style={{ margin: '5px 0' }}>
          <Paragraph>自由对话技能示例：</Paragraph>
          <ol>
            <li>
              您可以问我关于IT项目实施方面的问题，例如：“如何进行数据库的迁移，迁移时要注意哪些风险？”{' '}
            </li>
            <li>
              您可以问我关于大模型方面的问题，例如：“什么是大语言模型提示词，其作用是什么？”
            </li>
            <li>
              {' '}
              您可以问我关于IT知识方面的问题，例如：“请介绍一下ARM服务器和x86服务器,并对两者进行比较”
            </li>
          </ol>
        </div>

        <div style={{ margin: '5px 0' }}>
          <Paragraph>程序开发技能示例：</Paragraph>
          <ol>
            <li>在Linux环境中，如何统计当前目录下文件夹的大小？</li>
            <li>如何使用Ansible playbook 或如何配置 Linux服务器 yum 源？</li>
            <li>Python中如何定义一个函数？</li>
            <li>SQL语句中如何进行联合查询？</li>
          </ol>
        </div>

        <div style={{ margin: '5px 0' }}>
          <Paragraph>知识库技能示例：</Paragraph>
          <ol>
            <li>
              在大型模型和专业知识库的支持下，可以为您提供更专业的的IT运维支持和解决方案。
            </li>
            <li>
              请您选择相应领域的知识库，我将基于知识库的内容，为您提供更有针对性的建议和解决方案。
            </li>
            <li>
              比如有关达梦数据库，华为泰山服务器和Ansible自动化方面的支持。
            </li>
          </ol>
        </div>

        <div style={{ margin: '5px 0' }}>
          <Paragraph>CMDB技能示例：</Paragraph>
          <ol>
            <li>您可以问“北湖数据中心目前有多少设备”</li>
            <li>您可以问“ 按设备类型分组，统计每种设备类型的数量”</li>
            <li>您可以问“查询并统计设备类型为小型机在数据中心的分布情况”</li>
            <li>
              您可以问“查询并统计设备类型为ARM服务器采购时间的分布，以年为单位统计“
            </li>
          </ol>
        </div>

        <div style={{ margin: '5px 0' }}>
          <Paragraph>日志分析技能示例：</Paragraph>
          <ol>
            <li>
              请上传相关日志文件，我将解读日志内容，分析潜在问题，并尽力给出根本原因，解决方案和预防建议。
            </li>
            <li>
              无论您遇到什么问题，都请随时向我提问，我将竭尽全力为您提供支持和解决方案！
            </li>
          </ol>
        </div>
        <div style={{ margin: '5px 0' }}>
          <Paragraph>自动化运维示例：</Paragraph>
          <ol>
            <li>请列出所有技能库中的技能</li>
          </ol>
        </div>
      </Modal>
    </div>
  );
};

export default AssistantIntro;
