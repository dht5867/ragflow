import { Divider, Layout, Menu, theme, Button } from 'antd';
import React, { useState, useMemo } from 'react';
import { Outlet } from 'umi';
import '../locales/config';
import Header from './components/header';
import type { MenuProps } from 'antd';
import { useNavigateWithFromState } from '@/hooks/route-hook';

import styles from './index.less';
import {
  AppstoreOutlined,
  MessageOutlined,
  SearchOutlined,
  ToolOutlined,
  UserOutlined,
  TeamOutlined,
  DeploymentUnitOutlined
  
} from '@ant-design/icons';
import { useTranslate } from '@/hooks/common-hooks';
import { ReactComponent as FileIcon } from '@/assets/svg/file-management.svg';
import { ReactComponent as KnowledgeBaseIcon } from '@/assets/svg/knowledge-base.svg';
import { useLocation } from 'umi';

const { Content, Sider } = Layout;

const App: React.FC = () => {
  const {
    token: { colorBgContainer, borderRadiusLG, colorBgBase },
  } = theme.useToken(); // Added colorBgBase for background consistency
  const { pathname } = useLocation();
  const [collapsed, setCollapsed] = useState(false); // State to manage sidebar collapse
  type MenuItem = Required<MenuProps>['items'][number];
  const { t } = useTranslate('header');

  // Menu items array
  const items: MenuItem[] = useMemo(
    () => [
      {
        label: t('chat'),
        key: '/chat',
        icon: <MessageOutlined />,
      },
      {
        label: t('knowledgeBase'),
        key: '/knowledge',
        icon: <KnowledgeBaseIcon />,
      },
      {
        label: t('search'),
        key: '/search',
        icon: <SearchOutlined />,
      },
      {
        label: t('fileManager'),
        key: '/file',
        icon: <FileIcon />,
      },
      {
        label: t('automation'),
        key: '/automation',
        icon: <ToolOutlined />,
      },
      {
        label: t('flow'),
        key: '/flow',
        icon: <DeploymentUnitOutlined />,
      },
      {
        label: t('model'),
        key: '/user-setting/model',
        icon: <AppstoreOutlined />,
      },
      {
        label:  t('team'),
        key: '/user-setting/team',
        icon: <TeamOutlined  />,
      },
      {
        label: t('user'),
        key: '/user-setting/profile',
        icon: <UserOutlined />,
      },
    ],
    [t],
  );

  const currentKey = useMemo(() => {
    const matchedItem = items.find((item) => pathname.startsWith(item.key));
    return matchedItem ? matchedItem.key : '/chat'; // Default to '/chat' if no match
  }, [pathname, items]);
  

  const navigate = useNavigateWithFromState();

  // Handle menu item click
  const handleMenuClick: MenuProps['onClick'] = (e) => {
    navigate(e.key);
  };

  // Toggle collapsed state
  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  return (
    <Layout className={styles.layout}>
      <Header />
      <Divider orientationMargin={0} className={styles.divider} />
      <Layout>
        <Sider 
          width={200}
          theme="light"
          className={styles.sider}
          collapsible collapsed={collapsed} onCollapse={(value) => setCollapsed(value)} 
          style={{ background: colorBgBase, border: 1 }} // Set sidebar background color
        >
          {/* Toggle button for collapsing the sidebar */}
          {/* Vertical Menu in Sider */}
          <Menu
            onClick={handleMenuClick}
            selectedKeys={[currentKey]} // Current selected menu item
            theme="light"
            items={items}
            mode="inline"
            style={{ background: colorBgBase, border: 'none' }} // Ensure menu background color matches
          />
        </Sider>
        <Divider type="vertical" orientationMargin={0} className={styles.divider} />
        <Layout>
          <Content
            style={{
              background: colorBgContainer,
              borderRadius: borderRadiusLG,
              overflow: 'auto',
              display: 'flex',
            }}
          >
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default App;
