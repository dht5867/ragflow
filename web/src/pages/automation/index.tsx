import { useNavigateWithFromState } from '@/hooks/route-hook';
import React, { useEffect } from 'react';
import { useLocation } from 'umi';
import styles from './index.less'; // Add styles if needed
import { useTranslate } from '@/hooks/common-hooks';


const AutomationOps: React.FC = () => {
  const navigate = useNavigateWithFromState();
  const { pathname } = useLocation();
  const { t ,i18n} = useTranslate('chat');

  useEffect(() => {
    if (pathname !== '/automation') {
      navigate('/automation');
    }
  }, [pathname, navigate]);

  return (
    <div className={styles.iframeContainer}>
      <iframe
        src={`https://ansible.zerotrusts.cc/?lang=${i18n.language}`}
        title="自动化运维"
        width="100%"
        height="92%"
        style={{ border: 'none', overflow: 'hidden' }} // Style to remove iframe border
      />
    </div>
  );
};

export default AutomationOps;
