import { ReactComponent as FileIcon } from '@/assets/svg/file-management.svg';
import { ReactComponent as KnowledgeBaseIcon } from '@/assets/svg/knowledge-base.svg';
import { useTranslate } from '@/hooks/common-hooks';
import { useNavigateWithFromState } from '@/hooks/route-hook';
import { Layout, Menu, Space, theme } from 'antd';
import { useCallback, useMemo } from 'react';
import { useLocation } from 'umi';
import Toolbar from '../right-toolbar';

import { useFetchAppConf } from '@/hooks/logic-hooks';
import {
  AppstoreOutlined,
  MessageOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import styles from './index.less';

const { Header } = Layout;
type MenuItem = Required<MenuProps>['items'][number];

const RagHeader = () => {
  const {
    token: { colorBgContainer },
  } = theme.useToken();
  const navigate = useNavigateWithFromState();
  const { pathname } = useLocation();
  const { t } = useTranslate('header');
  const appConf = useFetchAppConf();

  // 将 tagsData 转换为 items 数组
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
      // New menu item for "自动化运维"
      {
        label: '自动化运维',
        key: '/automation', // The key used to identify this route
        icon: <AppstoreOutlined />, // You can replace this icon with a different one if needed
      },
    ],
    [t],
  );

  const handleLogoClick = useCallback(() => {
    navigate('/');
  }, [navigate]);

  // 当菜单项被点击时处理导航
  const handleMenuClick: MenuProps['onClick'] = (e) => {
    navigate(e.key); // 使用 key 作为路径导航
  };
  const currentKey = useMemo(() => {
    const matchedItem = items.find((item) => pathname.startsWith(item.key));
    return matchedItem ? matchedItem.key : '/chat'; // Default to '/chat' if no match
  }, [pathname, items]);
  
  return (
    <Header
      style={{
        padding: '0 16px',
        background: colorBgContainer,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        height: '72px',
        overflow: 'hidden', // 增加 overflow 防止元素溢出
      }}
    >
      <Space size={12} onClick={handleLogoClick} className={styles.logoWrapper}>
        <img src="/logo.jpg" alt="" className={styles.appIcon} />
        <span className={styles.appName}>{appConf.appName}</span>
      </Space>
      {/* 使用 Menu 组件实现导航 */}
      <Menu
        onClick={handleMenuClick}
        selectedKeys={[currentKey]} // 当前选中的菜单项
        mode="horizontal"
        items={items}
      />
      <Toolbar></Toolbar>
    </Header>
  );
};

export default RagHeader;
