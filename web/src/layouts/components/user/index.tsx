import React, { useState } from 'react';
import { Avatar, Dropdown, Menu, MenuProps, Modal } from 'antd';
import { useLogout } from '@/hooks/login-hooks';
import { useTranslate } from '@/hooks/common-hooks';

import { useFetchUserInfo } from '@/hooks/user-setting-hooks';

import styles from '../../index.less';
import UserSettingPassword from '@/pages/user-setting/setting-password';

const App: React.FC = () => {
  const { t } = useTranslate('header');
  const { data: userInfo } = useFetchUserInfo();
  const { logout } = useLogout();
  const [isModalVisible, setIsModalVisible] = useState(false);

  const showPasswordModal = () => {
    setIsModalVisible(true);
  };

  const handleLogout = () => {
    logout();
    // Optionally, you might want to redirect after logout
    // history.push('/login'); 
  };

  const handleMenuClick = (e: any) => {
    if (e.key === 'change-password') showPasswordModal();
    if (e.key === 'logout') handleLogout();
  };

  const menuItems: MenuProps['items'] = [
    {
      key: 'change-password',
      label: <span> {t('password')}</span>,
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      label: <span>{t('loginout')}</span>,
    },
  ];
  return (
    <>

      <Dropdown menu={{ items: menuItems, onClick: handleMenuClick }}>
        <Avatar
          size={32}
          className={styles.clickAvailable}
          src={
            userInfo.avatar ?? '/logout.png'
          }
        />
      </Dropdown>
      
      <Modal
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={null}
      >
        <UserSettingPassword />
      </Modal>
    </>
  );
};

export default App;
