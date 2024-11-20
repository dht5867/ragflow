import { ReactComponent as FileIcon } from '@/assets/svg/file-management.svg';
import { ReactComponent as KnowledgeBaseIcon } from '@/assets/svg/knowledge-base.svg';
import { useTranslate } from '@/hooks/common-hooks';
import { useNavigateWithFromState } from '@/hooks/route-hook';
import { MessageOutlined, SearchOutlined } from '@ant-design/icons';
import { Flex, Layout, Radio, Space, theme } from 'antd';
import { useCallback, useMemo } from 'react';
import { useLocation } from 'umi';
import Toolbar from '../right-toolbar';

import { useFetchAppConf } from '@/hooks/logic-hooks';
import {
  AppstoreOutlined,
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

  const currentPath = useMemo(() => {
    return (
      tagsData.find((x) => pathname.startsWith(x.path))?.name || 'knowledge'
    );
  }, [pathname, tagsData]);

  const handleChange = (path: string) => {
    navigate(path);
  };

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
        height: '48px',
        overflow: 'hidden', // 增加 overflow 防止元素溢出
      }}
    >
      <Space size={12} onClick={handleLogoClick} className={styles.logoWrapper}>
        <img src="/logo.svg" alt="" className={styles.appIcon} />
        <span className={styles.appName}>{appConf.appName}</span>
      </Space>
      <Space size={[0, 8]} wrap>
        <Radio.Group
          defaultValue="a"
          buttonStyle="solid"
          className={styles.radioGroup}
          value={currentPath}
        >
          {tagsData.map((item) => (
            <Radio.Button
              value={item.name}
              onClick={() => handleChange(item.path)}
              key={item.name}
            >
              <Flex align="center" gap={8}>
                <item.icon
                  className={styles.radioButtonIcon}
                  stroke={item.name === currentPath ? 'black' : 'white'}
                ></item.icon>
                {item.name}
              </Flex>
            </Radio.Button>
          ))}
        </Radio.Group>
      </Space>
      <Toolbar></Toolbar>
    </Header>
  );
};

export default RagHeader;
